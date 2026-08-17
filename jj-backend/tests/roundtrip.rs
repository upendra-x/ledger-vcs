//! Round-trip every object type through a running Ledger.
//!
//! This is the test that makes the crate more than a compiling type-check: jj's
//! own structures go in, Ledger's objects come out, and jj's structures come
//! back identical. Compiling against the trait proves the shape; only this
//! proves the mapping.
//!
//! Needs a Ledger. `LEDGER_URL` and `LEDGER_ENV` point at one — normally started
//! by `tests/e2e/test_jj_backend.py`, which is what runs this in CI. Without
//! them each test returns early rather than failing, because a Rust test that
//! only passes on a machine with a service running is a test that gets ignored.
//!
//! Returning early is indistinguishable from passing, which makes the skip its
//! own hazard: `cargo test` with no server reports "8 passed" having proved
//! nothing, and so does a harness whose server never came up. So a test that
//! really ran says so — see `ran()` — and the Python side counts the marks
//! rather than trusting the tally.

use futures::AsyncReadExt;
use jj_lib::backend::Backend;
use jj_lib::backend::ChangeId;
use jj_lib::backend::Commit;
use jj_lib::backend::MillisSinceEpoch;
use jj_lib::backend::Signature;
use jj_lib::backend::Timestamp;
use jj_lib::backend::Tree;
use jj_lib::backend::TreeValue;
use jj_lib::backend::CopyId;
use jj_lib::merge::Merge;
use jj_lib::object_id::ObjectId;
use jj_lib::repo_path::RepoPath;
use jj_lib::repo_path::RepoPathComponentBuf;
use ledger_jj_backend::LedgerBackend;

fn backend() -> Option<LedgerBackend> {
    let url = std::env::var("LEDGER_URL").ok()?;
    let env = std::env::var("LEDGER_ENV").ok()?;
    let token = std::env::var("LEDGER_TOKEN").ok();
    Some(LedgerBackend::connect(&url, &env, token).expect("could not reach Ledger"))
}

/// Evidence that a test body actually executed against a Ledger.
///
/// Printed rather than counted in process, because the check that matters is
/// made by the harness in another language: `cargo test`'s own "N passed" counts
/// the early returns too, so it cannot tell a suite that verified the mapping
/// from one that verified nothing at all.
///
/// Visible only under `--nocapture`, which is exactly how the harness runs it.
fn ran(what: &str) {
    println!("LEDGER_RAN {what}");
}

fn signature(name: &str) -> Signature {
    Signature {
        name: name.to_owned(),
        email: format!("{name}@example.invalid"),
        timestamp: Timestamp {
            timestamp: MillisSinceEpoch(1_700_000_000_000),
            tz_offset: 60,
        },
    }
}

#[tokio::test]
async fn it_reports_the_ids_jj_needs() {
    let Some(backend) = backend() else { return };
    ran("it_reports_the_ids_jj_needs");

    assert_eq!(backend.name(), "ledger");
    // Ledger names objects by BLAKE3-256 and change ids are 16 bytes; jj
    // validates every id it is handed against these.
    assert_eq!(backend.commit_id_length(), 32);
    assert_eq!(backend.change_id_length(), 16);
    assert_eq!(backend.empty_tree_id().as_bytes().len(), 32);
}

#[tokio::test]
async fn a_file_round_trips() {
    let Some(backend) = backend() else { return };
    ran("a_file_round_trips");
    let path = RepoPath::from_internal_string("data/train.bin").unwrap();
    let contents = b"the quick brown fox\n".repeat(500);

    let id = backend
        .write_file(path, &mut futures::io::Cursor::new(contents.clone()))
        .await
        .expect("write_file");

    let mut reader = backend.read_file(path, &id).await.expect("read_file");
    let mut read_back = Vec::new();
    reader.read_to_end(&mut read_back).await.unwrap();

    assert_eq!(read_back, contents);
}

#[tokio::test]
async fn identical_files_get_one_id() {
    // Deduplication, from jj's side: the name *is* the content hash, so two
    // writes of the same bytes cannot produce two objects.
    let Some(backend) = backend() else { return };
    ran("identical_files_get_one_id");
    let path = RepoPath::from_internal_string("a.txt").unwrap();

    let first = backend
        .write_file(path, &mut futures::io::Cursor::new(b"same".to_vec()))
        .await
        .unwrap();
    let second = backend
        .write_file(
            RepoPath::from_internal_string("elsewhere/b.txt").unwrap(),
            &mut futures::io::Cursor::new(b"same".to_vec()),
        )
        .await
        .unwrap();

    assert_eq!(first, second, "the same bytes got two names");
}

#[tokio::test]
async fn a_symlink_round_trips() {
    let Some(backend) = backend() else { return };
    ran("a_symlink_round_trips");
    let path = RepoPath::from_internal_string("link").unwrap();

    let id = backend.write_symlink(path, "../target").await.unwrap();
    assert_eq!(backend.read_symlink(path, &id).await.unwrap(), "../target");
}

#[tokio::test]
async fn a_tree_round_trips_with_its_modes() {
    let Some(backend) = backend() else { return };
    ran("a_tree_round_trips_with_its_modes");
    let root = RepoPath::root();

    let readme = backend
        .write_file(
            RepoPath::from_internal_string("README.md").unwrap(),
            &mut futures::io::Cursor::new(b"# demo\n".to_vec()),
        )
        .await
        .unwrap();
    let script = backend
        .write_file(
            RepoPath::from_internal_string("run.sh").unwrap(),
            &mut futures::io::Cursor::new(b"#!/bin/sh\necho hi\n".to_vec()),
        )
        .await
        .unwrap();
    let link = backend
        .write_symlink(RepoPath::from_internal_string("latest").unwrap(), "README.md")
        .await
        .unwrap();

    let tree = Tree::from_sorted_entries(vec![
        (
            RepoPathComponentBuf::new("README.md").unwrap(),
            TreeValue::File {
                id: readme.clone(),
                executable: false,
                copy_id: CopyId::placeholder(),
            },
        ),
        (
            RepoPathComponentBuf::new("latest").unwrap(),
            TreeValue::Symlink(link.clone()),
        ),
        (
            RepoPathComponentBuf::new("run.sh").unwrap(),
            TreeValue::File {
                id: script.clone(),
                executable: true,
                copy_id: CopyId::placeholder(),
            },
        ),
    ]);

    let id = backend.write_tree(root, &tree).await.expect("write_tree");
    let read_back = backend.read_tree(root, &id).await.expect("read_tree");

    let names: Vec<String> = read_back
        .names()
        .map(|n| n.as_internal_str().to_owned())
        .collect();
    assert_eq!(names, vec!["README.md", "latest", "run.sh"]);

    // The executable bit is inside the hash — a version that quietly dropped it
    // would not be the same version.
    match read_back
        .value(&RepoPathComponentBuf::new("run.sh").unwrap())
        .unwrap()
    {
        TreeValue::File {
            id, executable, ..
        } => {
            assert_eq!(id, &script);
            assert!(*executable, "the executable bit did not survive");
        }
        other => panic!("expected a file, got {other:?}"),
    }
    match read_back
        .value(&RepoPathComponentBuf::new("latest").unwrap())
        .unwrap()
    {
        TreeValue::Symlink(id) => assert_eq!(id, &link),
        other => panic!("expected a symlink, got {other:?}"),
    }
}

#[tokio::test]
async fn a_commit_round_trips_with_both_signatures() {
    // The lossy mapping, checked. jj carries two signatures with emails and
    // timezones; a Ledger commit carries one author string and one timestamp.
    // The rest rides in the commit's metadata, so this must come back *exactly*.
    let Some(backend) = backend() else { return };
    ran("a_commit_round_trips_with_both_signatures");

    let empty = backend.empty_tree_id().clone();
    let commit = Commit {
        parents: vec![backend.root_commit_id().clone()],
        predecessors: vec![],
        root_tree: Merge::resolved(empty),
        conflict_labels: Merge::resolved(String::new()),
        change_id: ChangeId::new(vec![0xab; 16]),
        description: String::from("teach the verifier about partial credit"),
        author: signature("agent-17"),
        committer: signature("ledger-ci"),
        secure_sig: None,
    };

    let (id, written) = backend.write_commit(commit.clone(), None).await.unwrap();
    let read_back = backend.read_commit(&id).await.unwrap();

    assert_eq!(read_back.description, commit.description);
    assert_eq!(read_back.change_id, commit.change_id);
    assert_eq!(read_back.author, commit.author, "author did not round-trip");
    assert_eq!(
        read_back.committer, commit.committer,
        "committer did not round-trip"
    );
    assert_eq!(read_back.parents, commit.parents);
    assert_eq!(read_back.root_tree, commit.root_tree);
    assert_eq!(written.description, commit.description);
}

#[tokio::test]
async fn a_conflicted_tree_is_refused_with_its_reason() {
    // The stated conflict gap. jj represents a conflicted tree as a
    // Merge with more than one term; Ledger has no such object, because its
    // merge refuses an overlap rather than storing one. The failure must name
    // itself rather than silently picking a side.
    let Some(backend) = backend() else { return };
    ran("a_conflicted_tree_is_refused_with_its_reason");

    let empty = backend.empty_tree_id().clone();
    let conflicted = Merge::from_vec(vec![empty.clone(), empty.clone(), empty.clone()]);
    let commit = Commit {
        parents: vec![backend.root_commit_id().clone()],
        predecessors: vec![],
        root_tree: conflicted,
        conflict_labels: Merge::resolved(String::new()),
        change_id: ChangeId::new(vec![0xcd; 16]),
        description: String::from("a conflicted state"),
        author: signature("agent-17"),
        committer: signature("agent-17"),
        secure_sig: None,
    };

    let error = backend
        .write_commit(commit, None)
        .await
        .expect_err("a conflicted tree must not be storable");
    let message = error.to_string();
    assert!(
        message.contains("conflict"),
        "the refusal should say why: {message}"
    );
}

#[tokio::test]
async fn signing_is_refused_rather_than_faked() {
    // A signature covers the backend's serialization, and Ledger's canonical
    // encoding is produced server-side. Signing here would need a second
    // implementation of it — the one thing this crate refuses to have.
    let Some(backend) = backend() else { return };
    ran("signing_is_refused_rather_than_faked");

    let empty = backend.empty_tree_id().clone();
    let commit = Commit {
        parents: vec![backend.root_commit_id().clone()],
        predecessors: vec![],
        root_tree: Merge::resolved(empty),
        conflict_labels: Merge::resolved(String::new()),
        change_id: ChangeId::new(vec![0xef; 16]),
        description: String::from("signed"),
        author: signature("agent-17"),
        committer: signature("agent-17"),
        secure_sig: None,
    };

    let mut sign = |_: &[u8]| Ok(vec![0u8; 8]);
    let error = backend
        .write_commit(commit, Some(&mut sign))
        .await
        .expect_err("signing must be refused, not faked");
    assert!(error.to_string().contains("sign"));
}
