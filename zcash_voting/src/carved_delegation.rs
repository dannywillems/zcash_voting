//! Restoring delegation state recovered by carving a wiped database.
//!
//! # Why this exists
//!
//! `clear_round` used to run before every `initRound`, to dodge a UNIQUE
//! constraint on retry. It cascades `bundles` away, and `bundles` holds
//! `van_comm_rand`, which was sampled from `OsRng` and writable through no
//! exported call. Reopening a poll therefore destroyed the only opening of a
//! VAN the chain already held. The wallet's client can now carve those secrets
//! back out of the freed pages and the write-ahead log, and had nowhere to put
//! them.
//!
//! # When this is still the only recourse
//!
//! Narrower than it was. Since 3.1.0, LOCAL voting-hotkey delegation derives
//! its VAN blinding from the stored hotkey secret and the exact round and
//! bundle identity, so restoring that secret reconstructs the same VAN with no
//! carving at all -- see `hotkey::VotingHotkey::stored_secret` and
//! `restored_hotkey_reconstructs_van_after_voting_database_loss`. Two cases
//! are left:
//!
//! 1. Rounds created before that change, whose blinding really was random.
//!    Those are the wiped devices this module exists for.
//! 2. `with_round_bound_voting_target`, which has no hotkey secret, so its
//!    blinding "remains randomly sampled and must be retained through the
//!    existing custody recovery material".
//!
//! Prefer the derivation wherever it applies. This is for the rounds it cannot
//! reach.
//!
//! # Why not `import_delegation_capability`
//!
//! That is a capability-TRANSFER protocol: canonical JSON, a digest
//! acknowledging delivered bytes to a funds controller, and validation against
//! a trusted voter context. Recovery has no controller and no delivered bytes.
//! It also rejects non-NULL heavy columns, and the affected rounds were
//! cleared *and rebuilt*, so those columns hold a different delegation's data
//! and the import reads as conflicting local state.
//!
//! # The resting state
//!
//! A restored bundle is left in exactly the shape an imported-capability
//! bundle has: the identifying tuple set, every heavy column NULL. Reusing
//! that state is deliberate -- nothing downstream has to learn a new shape,
//! and whatever already handles `round_has_imported_capability_bundles`
//! handles a restored round unchanged.

use rusqlite::{named_params, Connection};

use crate::error::VotingError;

/// One bundle's delegation state, as carved from a wiped database.
///
/// This is the tuple `import_delegation_capability` builds a bundle from,
/// minus the parts that are contextual (`wallet_id`) or fixed
/// (`address_index`, which is 0 for a version 1 voting hotkey).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CarvedBundle {
    pub bundle_index: u32,
    /// The blinding factor. The irreplaceable part: everything else in this
    /// struct is either public or re-derivable.
    pub van_comm_rand: [u8; 32],
    /// The VAN this blinding factor opens, published on chain.
    pub gov_comm: [u8; 32],
    /// Bundle weight in zatoshi.
    pub total_note_value: u64,
    /// The transaction that broadcast this delegation, when the carve found
    /// one.
    ///
    /// `None` is a legitimate result rather than a partial capture: the hash
    /// is stored only after submission returns, so its absence usually means
    /// nothing was broadcast -- in which case there was nothing to strand and
    /// re-delegating normally is the correct path, not a restore.
    pub delegation_tx_hash: Option<String>,
}

/// What a restore did, so a caller can log it without re-querying.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestoreOutcome {
    /// Bundles whose secrets this call wrote.
    pub restored: u32,
    /// Bundles that already held the carved secrets, so nothing was written.
    pub already_present: u32,
}

impl RestoreOutcome {
    /// Whether the database was left unchanged.
    pub fn is_noop(&self) -> bool {
        self.restored == 0
    }
}

fn invalid(message: impl Into<String>) -> VotingError {
    VotingError::InvalidInput {
        message: message.into(),
    }
}

fn internal(message: impl Into<String>) -> VotingError {
    VotingError::Internal {
        message: message.into(),
    }
}

/// Restores carved delegation state over a round's bundles, atomically.
///
/// # Invariants
///
/// 1. **Atomic.** Every bundle or none; the caller supplies the transaction.
/// 2. **Idempotent.** A bundle already holding the carved `van_comm_rand` is
///    counted and skipped. Recovery runs on every cold launch, so a restore
///    that was not a no-op on the second run would rewrite state forever.
/// 3. **Never destroys a different delegation.** A bundle holding a
///    *different* secret AND a `delegation_tx_hash` of its own is a second
///    broadcast, not a rebuild. Choosing between two broadcast delegations is
///    not this function's decision, so it refuses.
/// 4. **Clears the superseded generation.** The rebuilt heavy columns describe
///    a different PCZT and cannot be opened by the restored VAN. They are set
///    NULL rather than left, so no row ever mixes two generations.
/// 5. **Does not invent confirmation state.** `van_leaf_position` is cleared,
///    not guessed; tree sync re-derives it from the chain.
/// 6. **Will not create rows.** A round or bundle that is not there is an
///    error, not something to conjure.
pub fn restore_carved_delegation(
    conn: &Connection,
    round_id: &str,
    wallet_id: &str,
    bundles: &[CarvedBundle],
) -> Result<RestoreOutcome, VotingError> {
    if bundles.is_empty() {
        return Err(invalid("no carved bundles to restore"));
    }

    let mut indices: Vec<u32> = bundles.iter().map(|b| b.bundle_index).collect();
    indices.sort_unstable();
    indices.dedup();
    if indices.len() != bundles.len() {
        return Err(invalid("carved bundles contain a duplicate bundle_index"));
    }

    let round_exists: bool = conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM rounds
                           WHERE round_id = :round_id AND wallet_id = :wallet_id)",
            named_params! { ":round_id": round_id, ":wallet_id": wallet_id },
            |row| row.get(0),
        )
        .map_err(|e| internal(format!("failed to look up round: {e}")))?;
    if !round_exists {
        return Err(invalid(format!(
            "round {round_id} does not exist for this wallet; restore does not create rounds"
        )));
    }

    let mut outcome = RestoreOutcome {
        restored: 0,
        already_present: 0,
    };

    for bundle in bundles {
        let existing: (Option<Vec<u8>>, Option<String>) = conn
            .query_row(
                "SELECT van_comm_rand, delegation_tx_hash FROM bundles
                 WHERE round_id = :round_id AND wallet_id = :wallet_id
                   AND bundle_index = :bundle_index",
                named_params! {
                    ":round_id": round_id,
                    ":wallet_id": wallet_id,
                    ":bundle_index": bundle.bundle_index as i64,
                },
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => invalid(format!(
                    "bundle {} of round {round_id} does not exist; restore does not create bundles",
                    bundle.bundle_index
                )),
                other => internal(format!("failed to read bundle: {other}")),
            })?;

        let (stored_rand, stored_hash) = existing;

        if stored_rand.as_deref() == Some(&bundle.van_comm_rand[..]) {
            // Invariant 2. Already the carved secret; leave the row alone
            // entirely, including whatever confirmation state it has since
            // accumulated.
            outcome.already_present += 1;
            continue;
        }

        // Invariant 3. A different secret that was itself broadcast is not a
        // rebuild to be overwritten.
        if let (Some(stored), Some(hash)) = (stored_rand.as_ref(), stored_hash.as_ref()) {
            let differs = stored.as_slice() != bundle.van_comm_rand;
            let same_tx = bundle.delegation_tx_hash.as_deref() == Some(hash.as_str());
            if differs && !same_tx {
                return Err(invalid(format!(
                    "bundle {} of round {round_id} holds a different delegation that was \
                     already broadcast (tx {hash}); refusing to overwrite it",
                    bundle.bundle_index
                )));
            }
        }

        // Invariants 4 and 5: write the carved tuple and clear everything that
        // belonged to the superseded generation, so the row lands in the
        // imported-capability shape rather than a mixture of the two.
        let rows = conn
            .execute(
                "UPDATE bundles SET
                     van_comm_rand = :rand,
                     gov_comm = :van,
                     total_note_value = :total,
                     address_index = 0,
                     delegation_tx_hash = :tx_hash,
                     note_positions_blob = NULL,
                     note_identity_hashes_blob = NULL,
                     dummy_nullifiers = NULL,
                     rho_signed = NULL,
                     padded_note_data = NULL,
                     nf_signed = NULL,
                     cmx_new = NULL,
                     alpha = NULL,
                     rseed_signed = NULL,
                     rseed_output = NULL,
                     van_leaf_position = NULL,
                     rk = NULL,
                     gov_nullifiers_blob = NULL,
                     padded_note_secrets = NULL,
                     pczt_sighash = NULL,
                     tx1_effects = NULL
                 WHERE round_id = :round_id AND wallet_id = :wallet_id
                   AND bundle_index = :bundle_index",
                named_params! {
                    ":rand": &bundle.van_comm_rand[..],
                    ":van": &bundle.gov_comm[..],
                    ":total": bundle.total_note_value as i64,
                    ":tx_hash": bundle.delegation_tx_hash.as_deref(),
                    ":round_id": round_id,
                    ":wallet_id": wallet_id,
                    ":bundle_index": bundle.bundle_index as i64,
                },
            )
            .map_err(|e| internal(format!("failed to restore carved bundle: {e}")))?;

        if rows != 1 {
            return Err(internal(format!(
                "restoring bundle {} of round {round_id} touched {rows} rows, expected 1",
                bundle.bundle_index
            )));
        }
        outcome.restored += 1;
    }

    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::VotingDb;

    const ROUND: &str = "aa";
    const WALLET: &str = "w1";

    /// A round with one bundle holding a REBUILT delegation: a different
    /// secret from the carved one, plus the heavy columns the rebuild wrote.
    /// This is the state an affected device is in.
    fn rebuilt_database() -> VotingDb {
        let db = VotingDb::open(":memory:").unwrap();
        let conn = db.conn();
        conn.execute(
            "INSERT INTO rounds (
                 round_id, wallet_id, network, snapshot_height,
                 ea_pk, nc_root, nullifier_imt_root, created_at
             ) VALUES (?1, ?2, 'mainnet', 1000, ?3, ?4, ?5, 0)",
            rusqlite::params![
                ROUND,
                WALLET,
                &[0xEAu8; 32][..],
                &[0xA1u8; 32][..],
                &[0xB1u8; 32][..]
            ],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO bundles (
                 round_id, wallet_id, bundle_index, van_comm_rand, gov_comm,
                 total_note_value, alpha, rk, pczt_sighash
             ) VALUES (?1, ?2, 0, ?3, ?4, 5, ?5, ?6, ?7)",
            rusqlite::params![
                ROUND,
                WALLET,
                &[0xBBu8; 32][..],
                &[0xCCu8; 32][..],
                &[0x11u8; 32][..],
                &[0x22u8; 32][..],
                &[0x33u8; 32][..]
            ],
        )
        .unwrap();
        drop(conn);
        db
    }

    fn carved() -> CarvedBundle {
        CarvedBundle {
            bundle_index: 0,
            van_comm_rand: [0xAA; 32],
            gov_comm: [0xDD; 32],
            total_note_value: 42,
            delegation_tx_hash: Some("d0".repeat(32)),
        }
    }

    fn column_is_null(conn: &Connection, name: &str) -> bool {
        conn.query_row(
            &format!("SELECT {name} IS NULL FROM bundles WHERE round_id = ?1"),
            rusqlite::params![ROUND],
            |row| row.get::<_, bool>(0),
        )
        .unwrap()
    }

    /// The whole point: the broadcast secret replaces the rebuild's.
    #[test]
    fn restores_the_carved_secret_over_a_rebuilt_bundle() {
        let db = rebuilt_database();
        let conn = db.conn();
        let outcome = restore_carved_delegation(&conn, ROUND, WALLET, &[carved()]).unwrap();

        assert_eq!(outcome.restored, 1);
        assert_eq!(outcome.already_present, 0);

        let (rand, van, total, hash): (Vec<u8>, Vec<u8>, i64, String) = conn
            .query_row(
                "SELECT van_comm_rand, gov_comm, total_note_value, delegation_tx_hash
                 FROM bundles WHERE round_id = ?1",
                rusqlite::params![ROUND],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )
            .unwrap();
        assert_eq!(rand, vec![0xAA; 32]);
        assert_eq!(van, vec![0xDD; 32]);
        assert_eq!(total, 42);
        assert_eq!(hash, "d0".repeat(32));
    }

    /// Invariant 4. The rebuild's heavy columns describe a different PCZT and
    /// cannot be opened by the restored VAN, so none may survive.
    #[test]
    fn clears_every_column_belonging_to_the_superseded_generation() {
        let db = rebuilt_database();
        let conn = db.conn();
        restore_carved_delegation(&conn, ROUND, WALLET, &[carved()]).unwrap();

        for column in [
            "alpha",
            "rk",
            "pczt_sighash",
            "nf_signed",
            "cmx_new",
            "rho_signed",
            "rseed_signed",
            "rseed_output",
            "dummy_nullifiers",
            "padded_note_data",
            "padded_note_secrets",
            "gov_nullifiers_blob",
            "tx1_effects",
            "note_positions_blob",
            "note_identity_hashes_blob",
        ] {
            assert!(
                column_is_null(&conn, column),
                "{column} survived the restore"
            );
        }
    }

    /// Invariant 5. Confirmation state is re-derived by tree sync, never
    /// guessed here.
    #[test]
    fn does_not_invent_a_van_leaf_position() {
        let db = rebuilt_database();
        let conn = db.conn();
        restore_carved_delegation(&conn, ROUND, WALLET, &[carved()]).unwrap();
        assert!(column_is_null(&conn, "van_leaf_position"));
    }

    /// Invariant 2. Recovery runs on every cold launch, so the second run must
    /// change nothing.
    #[test]
    fn is_idempotent_across_repeated_recoveries() {
        let db = rebuilt_database();
        let conn = db.conn();
        let first = restore_carved_delegation(&conn, ROUND, WALLET, &[carved()]).unwrap();
        let second = restore_carved_delegation(&conn, ROUND, WALLET, &[carved()]).unwrap();

        assert_eq!(first.restored, 1);
        assert!(!first.is_noop());
        assert_eq!(second.restored, 0);
        assert_eq!(second.already_present, 1);
        assert!(second.is_noop());
    }

    /// Invariant 3. A different secret that was itself broadcast is a second
    /// delegation, not a rebuild; picking between them is not ours to do.
    #[test]
    fn refuses_to_overwrite_a_different_delegation_that_was_broadcast() {
        let db = rebuilt_database();
        let conn = db.conn();
        conn.execute(
            "UPDATE bundles SET delegation_tx_hash = ?1 WHERE round_id = ?2",
            rusqlite::params!["ee".repeat(32), ROUND],
        )
        .unwrap();

        let err = restore_carved_delegation(&conn, ROUND, WALLET, &[carved()]).unwrap_err();
        assert!(
            format!("{err:?}").contains("already broadcast"),
            "unexpected error: {err:?}"
        );

        // And it really did leave the row alone.
        let rand: Vec<u8> = conn
            .query_row(
                "SELECT van_comm_rand FROM bundles WHERE round_id = ?1",
                rusqlite::params![ROUND],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            rand,
            vec![0xBB; 32],
            "the stored delegation was overwritten"
        );
    }

    /// Invariant 6. Restore repairs rounds; it does not conjure them.
    #[test]
    fn refuses_a_round_that_does_not_exist() {
        let db = rebuilt_database();
        let conn = db.conn();
        let err = restore_carved_delegation(&conn, "ffff", WALLET, &[carved()]).unwrap_err();
        assert!(format!("{err:?}").contains("does not exist"), "{err:?}");
    }

    /// Invariant 6, for the bundle. A carve that names a bundle the round does
    /// not have is a bug in the caller, not a row to create.
    #[test]
    fn refuses_a_bundle_that_does_not_exist() {
        let db = rebuilt_database();
        let conn = db.conn();
        let mut bundle = carved();
        bundle.bundle_index = 7;
        let err = restore_carved_delegation(&conn, ROUND, WALLET, &[bundle]).unwrap_err();
        assert!(format!("{err:?}").contains("does not exist"), "{err:?}");
    }

    /// A carve that names one bundle twice is ambiguous about which secret
    /// wins, so it is rejected rather than resolved by input order.
    #[test]
    fn rejects_duplicate_bundle_indices() {
        let db = rebuilt_database();
        let conn = db.conn();
        let err =
            restore_carved_delegation(&conn, ROUND, WALLET, &[carved(), carved()]).unwrap_err();
        assert!(format!("{err:?}").contains("duplicate"), "{err:?}");
    }

    #[test]
    fn rejects_an_empty_restore() {
        let db = rebuilt_database();
        let conn = db.conn();
        let err = restore_carved_delegation(&conn, ROUND, WALLET, &[]).unwrap_err();
        assert!(format!("{err:?}").contains("no carved bundles"), "{err:?}");
    }

    /// A delegation that was never broadcast carries no hash, and that must
    /// restore cleanly rather than being treated as incomplete.
    #[test]
    fn restores_a_bundle_that_was_never_broadcast() {
        let db = rebuilt_database();
        let conn = db.conn();
        let mut bundle = carved();
        bundle.delegation_tx_hash = None;

        let outcome = restore_carved_delegation(&conn, ROUND, WALLET, &[bundle]).unwrap();

        assert_eq!(outcome.restored, 1);
        assert!(column_is_null(&conn, "delegation_tx_hash"));
    }

    /// Invariant 1. A failure part-way leaves nothing behind, provided the
    /// caller supplies the transaction -- which is what this asserts.
    #[test]
    fn a_rejected_bundle_rolls_back_the_ones_before_it() {
        let db = rebuilt_database();
        let mut conn = db.conn();
        conn.execute(
            "INSERT INTO bundles (round_id, wallet_id, bundle_index, van_comm_rand)
             VALUES (?1, ?2, 1, ?3)",
            rusqlite::params![ROUND, WALLET, &[0xEEu8; 32][..]],
        )
        .unwrap();

        let mut second = carved();
        second.bundle_index = 9; // does not exist, so the call fails on it

        let tx = conn.transaction().unwrap();
        let result = restore_carved_delegation(&tx, ROUND, WALLET, &[carved(), second]);
        assert!(result.is_err());
        drop(tx); // rolls back

        let rand: Vec<u8> = conn
            .query_row(
                "SELECT van_comm_rand FROM bundles WHERE round_id = ?1 AND bundle_index = 0",
                rusqlite::params![ROUND],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(rand, vec![0xBB; 32], "bundle 0 was not rolled back");
    }
}
