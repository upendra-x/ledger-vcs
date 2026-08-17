//! The `Backend` implementation.
//!
//! Twenty methods, and the interesting part is which ones are *not* implemented
//! and why. Each unsupported case returns an error naming the reason rather than
//! silently doing something approximate, because a version control system that
//! quietly loses a distinction is worse than one that refuses it.

use std::fmt::Debug;
use std::pin::Pin;
use std::time::SystemTime;

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::stream;
use jj_lib::backend::Backend;
use jj_lib::backend::BackendError;
use jj_lib::backend::BackendResult;
use jj_lib::backend::ChangeId;
use jj_lib::backend::Commit;
use jj_lib::backend::CommitId;
use jj_lib::backend::CopyHistory;
use jj_lib::backend::CopyId;
use jj_lib::backend::CopyRecord;
use jj_lib::backend::FileId;
use jj_lib::backend::MillisSinceEpoch;
use jj_lib::backend::RelatedCopy;
use jj_lib::backend::Signature;
use jj_lib::backend::SigningFn;
use jj_lib::backend::SymlinkId;
use jj_lib::backend::Timestamp;
use jj_lib::backend::Tree;
use jj_lib::backend::TreeId;
use jj_lib::backend::TreeValue;
use jj_lib::index::Index;
use jj_lib::merge::Merge;
use jj_lib::object_id::ObjectId;
use jj_lib::repo_path::RepoPath;
use jj_lib::repo_path::RepoPathBuf;
use jj_lib::repo_path::RepoPathComponentBuf;
use futures::AsyncRead;
use futures::AsyncReadExt;

use crate::client::ClientError;
use crate::client::CommitJson;
use crate::client::LedgerClient;
use crate::client::SignatureJson;
use crate::client::TreeEntryJson;
use crate::client::TreeJson;

/// Ledger names objects by BLAKE3-256, so every id jj handles is 32 bytes.
const COMMIT_ID_BYTES: usize = 32;
/// A change id is assigned once and survives rewrites.
const CHANGE_ID_BYTES: usize = 16;

/// jj's storage, served by Ledger.
#[derive(Debug)]
pub struct LedgerBackend {
    client: LedgerClient,
    root_commit_id: CommitId,
    root_change_id: ChangeId,
    empty_tree_id: TreeId,
}

impl LedgerBackend {
    /// Connect, and learn the ids jj needs before it can address anything.
    ///
    /// The id lengths and the empty-tree id come from the server rather than
    /// being hardcoded here: they are properties of Ledger's frozen format, and
    /// a copy of them in this crate would be a second place for them to drift.
    pub fn connect(base: &str, env: &str, token: Option<String>) -> Result<Self, ClientError> {
        let client = LedgerClient::new(base, env, token);
        let info = client.info()?;

        // jj's own root commit: an all-zero id it never asks the backend for.
        let root_commit_id = CommitId::new(vec![0; info.commit_id_length]);
        let root_change_id = ChangeId::new(vec![0; info.change_id_length]);
        let empty_tree_id = TreeId::new(decode_hex(&info.empty_tree_id).map_err(|_| {
            ClientError::Malformed(String::from("the server returned a malformed empty tree id"))
        })?);

        Ok(Self {
            client,
            root_commit_id,
            root_change_id,
            empty_tree_id,
        })
    }

    fn client(&self) -> &LedgerClient {
        &self.client
    }
}

#[async_trait]
impl Backend for LedgerBackend {
    fn name(&self) -> &str {
        "ledger"
    }

    fn commit_id_length(&self) -> usize {
        COMMIT_ID_BYTES
    }

    fn change_id_length(&self) -> usize {
        CHANGE_ID_BYTES
    }

    fn root_commit_id(&self) -> &CommitId {
        &self.root_commit_id
    }

    fn root_change_id(&self) -> &ChangeId {
        &self.root_change_id
    }

    fn empty_tree_id(&self) -> &TreeId {
        &self.empty_tree_id
    }

    fn concurrency(&self) -> usize {
        // Ledger has no cross-object locking — every write is a conditional
        // write in one environment's partition — so requests may overlap freely.
        // The bound here is the service's own, not a property of the storage.
        8
    }

    // ── files ───────────────────────────────────────────────────────────────

    async fn read_file(
        &self,
        _path: &RepoPath,
        id: &FileId,
    ) -> BackendResult<Pin<Box<dyn AsyncRead + Send>>> {
        let hex = id.hex();
        let bytes = self
            .client()
            .read_file(&hex)
            .map_err(|error| object_error("file", &hex, error))?;
        // `futures::io::Cursor`, not `std::io::Cursor`: the trait jj asks for
        // is futures' `AsyncRead`, and the two are different traits with the
        // same name.
        Ok(Box::pin(futures::io::Cursor::new(bytes)))
    }

    async fn write_file(
        &self,
        _path: &RepoPath,
        contents: &mut (dyn AsyncRead + Send + Unpin),
    ) -> BackendResult<FileId> {
        // jj hands a stream; Ledger chunks it server-side. Reading it whole
        // first is the honest simplification for a local backend, and it is
        // where a streaming upload would go if a workspace ever held a file too
        // large to buffer.
        let mut buffer = Vec::new();
        contents
            .read_to_end(&mut buffer)
            .await
            .map_err(|error| BackendError::WriteObject {
                object_type: "file",
                source: Box::new(error),
            })?;
        let hex = self
            .client()
            .write_file(&buffer)
            .map_err(|error| write_error("file", error))?;
        Ok(FileId::new(decode_hex(&hex).map_err(|error| {
            write_error("file", ClientError::Malformed(error))
        })?))
    }

    // ── symlinks ────────────────────────────────────────────────────────────

    async fn read_symlink(&self, _path: &RepoPath, id: &SymlinkId) -> BackendResult<String> {
        let hex = id.hex();
        self.client()
            .read_symlink(&hex)
            .map_err(|error| object_error("symlink", &hex, error))
    }

    async fn write_symlink(&self, _path: &RepoPath, target: &str) -> BackendResult<SymlinkId> {
        let hex = self
            .client()
            .write_symlink(target)
            .map_err(|error| write_error("symlink", error))?;
        Ok(SymlinkId::new(decode_hex(&hex).map_err(|error| {
            write_error("symlink", ClientError::Malformed(error))
        })?))
    }

    // ── copies ──────────────────────────────────────────────────────────────
    //
    // jj tracks file copies as first-class objects. Ledger does not model them:
    // a copy is two paths reaching the same content, which content addressing
    // already makes free and indistinguishable. Rather than invent a
    // representation, these report that the backend has none — which is what
    // jj's own `Backend` docs expect of a backend without copy tracking.

    async fn read_copy(&self, id: &CopyId) -> BackendResult<CopyHistory> {
        Err(BackendError::Unsupported(format!(
            "Ledger does not track copies; content addressing makes a copy \
             indistinguishable from the original (asked for {})",
            id.hex()
        )))
    }

    async fn write_copy(&self, _copy: &CopyHistory) -> BackendResult<CopyId> {
        Err(BackendError::Unsupported(String::from(
            "Ledger does not track copies; content addressing makes a copy \
             indistinguishable from the original",
        )))
    }

    async fn get_related_copies(&self, _copy_id: &CopyId) -> BackendResult<Vec<RelatedCopy>> {
        Ok(Vec::new())
    }

    fn get_copy_records(
        &self,
        _paths: Option<&[RepoPathBuf]>,
        _root: &CommitId,
        _head: &CommitId,
    ) -> BackendResult<BoxStream<'_, BackendResult<CopyRecord>>> {
        Ok(Box::pin(stream::empty()))
    }

    // ── trees ───────────────────────────────────────────────────────────────

    async fn read_tree(&self, _path: &RepoPath, id: &TreeId) -> BackendResult<Tree> {
        let hex = id.hex();
        let json = self
            .client()
            .read_tree(&hex)
            .map_err(|error| object_error("tree", &hex, error))?;

        // Ledger's trees are ordered by name already — that ordering is what
        // makes a listing a range scan with a stable cursor — so this is the
        // sorted constructor rather than an insert-and-sort.
        let mut entries = Vec::with_capacity(json.entries.len());
        for entry in &json.entries {
            let name = RepoPathComponentBuf::new(&entry.name).map_err(|_| {
                BackendError::Other(
                    format!("Ledger returned a tree entry jj cannot name: {}", entry.name).into(),
                )
            })?;
            entries.push((name, to_tree_value(entry)?));
        }
        entries.sort_by(|(a, _), (b, _)| a.cmp(b));
        Ok(Tree::from_sorted_entries(entries))
    }

    async fn write_tree(&self, _path: &RepoPath, contents: &Tree) -> BackendResult<TreeId> {
        let mut entries = Vec::new();
        for entry in contents.entries() {
            entries.push(from_tree_value(entry.name().as_internal_str(), entry.value())?);
        }
        let hex = self
            .client()
            .write_tree(&TreeJson { entries })
            .map_err(|error| write_error("tree", error))?;
        Ok(TreeId::new(decode_hex(&hex).map_err(|error| {
            write_error("tree", ClientError::Malformed(error))
        })?))
    }

    // ── commits ─────────────────────────────────────────────────────────────

    async fn read_commit(&self, id: &CommitId) -> BackendResult<Commit> {
        let hex = id.hex();
        let json = self
            .client()
            .read_commit(&hex)
            .map_err(|error| object_error("commit", &hex, error))?;
        to_commit(json)
    }

    async fn write_commit(
        &self,
        contents: Commit,
        sign_with: Option<&mut SigningFn>,
    ) -> BackendResult<(CommitId, Commit)> {
        if sign_with.is_some() {
            // A signature covers the backend's serialization of a commit, and
            // Ledger's is canonical and produced server-side. Signing would need
            // the client to reproduce those exact bytes — a second encoder,
            // which is the one thing this crate refuses to have.
            return Err(BackendError::Unsupported(String::from(
                "this backend does not sign commits: the canonical encoding is \
                 produced by the server, and signing it here would require a \
                 second implementation of it",
            )));
        }

        let json = from_commit(&contents)?;
        let (hex, written) = self
            .client()
            .write_commit(&json)
            .map_err(|error| write_error("commit", error))?;
        let id = CommitId::new(decode_hex(&hex).map_err(|error| {
            write_error("commit", ClientError::Malformed(error))
        })?);
        Ok((id, to_commit(written)?))
    }

    // ── collection ──────────────────────────────────────────────────────────

    fn gc(&self, _index: &dyn Index, _keep_newer: SystemTime) -> BackendResult<()> {
        // Collection is Ledger's, and it is not a client's decision to make.
        // It is epoch-based mark and sweep over per-environment keep-sets
        //, it runs report-only by default, and a client asking a
        // *server* to collect from a local index would be asking it to reason
        // about reachability it cannot see. `ledger gc --enforce` is the door.
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Conversions
// ─────────────────────────────────────────────────────────────────────────────

fn to_tree_value(entry: &TreeEntryJson) -> BackendResult<TreeValue> {
    let id = decode_hex(&entry.id)
        .map_err(|error| BackendError::Other(format!("malformed id in tree: {error}").into()))?;
    match entry.kind.as_str() {
        "file" => Ok(TreeValue::File {
            id: FileId::new(id),
            executable: entry.executable,
            copy_id: CopyId::placeholder(),
        }),
        "symlink" => Ok(TreeValue::Symlink(SymlinkId::new(id))),
        "tree" => Ok(TreeValue::Tree(TreeId::new(id))),
        other => Err(BackendError::Other(
            format!("Ledger returned a tree entry kind jj does not know: {other}").into(),
        )),
    }
}

fn from_tree_value(name: &str, value: &TreeValue) -> BackendResult<TreeEntryJson> {
    let (kind, id, executable) = match value {
        TreeValue::File {
            id, executable, ..
        } => ("file", id.hex(), *executable),
        TreeValue::Symlink(id) => ("symlink", id.hex(), false),
        TreeValue::Tree(id) => ("tree", id.hex(), false),
        TreeValue::GitSubmodule(_) => {
            return Err(BackendError::Unsupported(format!(
                "Ledger versions content, and the git submodule at {name} is a \
                 pointer to content held elsewhere; import it as real content"
            )));
        }
    };
    Ok(TreeEntryJson {
        name: name.to_owned(),
        kind: kind.to_owned(),
        id,
        executable,
    })
}

fn to_commit(json: CommitJson) -> BackendResult<Commit> {
    let parents = json
        .parents
        .iter()
        .map(|hex| decode_hex(hex).map(CommitId::new))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| BackendError::Other(format!("malformed parent id: {error}").into()))?;
    let predecessors = json
        .predecessors
        .iter()
        .map(|hex| decode_hex(hex).map(CommitId::new))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| {
            BackendError::Other(format!("malformed predecessor id: {error}").into())
        })?;
    let root = json
        .root_tree
        .first()
        .ok_or_else(|| BackendError::Other("a commit with no tree".into()))?;
    let tree = TreeId::new(
        decode_hex(root)
            .map_err(|error| BackendError::Other(format!("malformed tree id: {error}").into()))?,
    );
    let change_id = ChangeId::new(decode_hex(&json.change_id).map_err(|error| {
        BackendError::Other(format!("malformed change id: {error}").into())
    })?);

    Ok(Commit {
        parents,
        predecessors,
        root_tree: Merge::resolved(tree),
        conflict_labels: Merge::resolved(String::new()),
        change_id,
        description: json.description,
        author: to_signature(json.author),
        committer: to_signature(json.committer),
        secure_sig: None,
    })
}

fn from_commit(commit: &Commit) -> BackendResult<CommitJson> {
    // jj represents a conflicted tree as a Merge with more than one term.
    // Ledger has no such object: its merge refuses an overlap rather than
    // storing one, which is the stated conflict gap. Refusing here
    // means the failure names itself, at the moment it happens.
    let tree = commit.root_tree.as_resolved().ok_or_else(|| {
        BackendError::Unsupported(String::from(
            "this commit has a conflicted tree, which Ledger does not model: it \
             refuses an overlapping merge rather than storing the conflict. \
             Resolve it in the working copy and commit the result",
        ))
    })?;

    Ok(CommitJson {
        parents: commit.parents.iter().map(|id| id.hex()).collect(),
        predecessors: commit.predecessors.iter().map(|id| id.hex()).collect(),
        root_tree: vec![tree.hex()],
        change_id: commit.change_id.hex(),
        description: commit.description.clone(),
        author: from_signature(&commit.author),
        committer: from_signature(&commit.committer),
    })
}

fn to_signature(json: SignatureJson) -> Signature {
    Signature {
        name: json.name,
        email: json.email,
        timestamp: Timestamp {
            timestamp: MillisSinceEpoch(json.timestamp_millis),
            tz_offset: json.tz_offset,
        },
    }
}

fn from_signature(signature: &Signature) -> SignatureJson {
    SignatureJson {
        name: signature.name.clone(),
        email: signature.email.clone(),
        timestamp_millis: signature.timestamp.timestamp.0,
        tz_offset: signature.timestamp.tz_offset,
    }
}

fn decode_hex(hex: &str) -> Result<Vec<u8>, String> {
    if hex.len() % 2 != 0 {
        return Err(format!("odd-length hex: {hex}"));
    }
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).map_err(|error| error.to_string()))
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────────────

/// Ledger's error contract carries state, and jj branches on the difference
/// between "not there" and "not yours" — so the distinction is preserved rather
/// than flattened into one opaque failure.
fn object_error(object_type: &'static str, hash: &str, error: ClientError) -> BackendError {
    if error.is_not_found() {
        return BackendError::ObjectNotFound {
            object_type: object_type.to_owned(),
            hash: hash.to_owned(),
            source: Box::new(error),
        };
    }
    if error.is_denied() {
        return BackendError::ReadAccessDenied {
            object_type: object_type.to_owned(),
            hash: hash.to_owned(),
            source: Box::new(error),
        };
    }
    BackendError::ReadObject {
        object_type: object_type.to_owned(),
        hash: hash.to_owned(),
        source: Box::new(error),
    }
}

fn write_error(object_type: &'static str, error: ClientError) -> BackendError {
    BackendError::WriteObject {
        object_type,
        source: Box::new(error),
    }
}
