//! The HTTP client, and nothing else.
//!
//! Deliberately thin. It does not know what a commit is — it moves JSON and
//! bytes to and from Ledger's jj adapter, and every decision about what those
//! mean lives one layer up in [`crate::backend`].
//!
//! **It never hashes anything.** Object names come back from the server, which
//! is the whole point: a second implementation of the canonical encoding, in
//! another language and maintained separately, is exactly the silent-divergence
//! failure Ledger's frozen format exists to prevent. Two encoders that agree
//! today and disagree after one careless edit would fork the corpus, and nothing
//! would report an error.

use std::time::Duration;

use serde::Deserialize;
use serde::Serialize;

/// How long a call to a Ledger on the same machine may take before it is a
/// problem rather than a delay.
const TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("ledger is unreachable: {0}")]
    Unreachable(String),
    #[error("ledger returned {status}: {message}")]
    Status { status: u16, message: String },
    #[error("ledger returned a body this backend cannot read: {0}")]
    Malformed(String),
}

/// The subset of Ledger's error contract this backend distinguishes.
impl ClientError {
    pub fn is_not_found(&self) -> bool {
        matches!(self, ClientError::Status { status: 404, .. })
    }

    pub fn is_denied(&self) -> bool {
        matches!(self, ClientError::Status { status: 401 | 403, .. })
    }
}

pub type ClientResult<T> = Result<T, ClientError>;

#[derive(Debug, Clone, Deserialize)]
pub struct BackendInfo {
    pub name: String,
    pub commit_id_length: usize,
    pub change_id_length: usize,
    pub empty_tree_id: String,
    pub format_fingerprint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeEntryJson {
    pub name: String,
    /// `file`, `symlink` or `tree`. A git submodule is refused by the server,
    /// because Ledger versions content and a submodule points at content held
    /// somewhere else.
    pub kind: String,
    pub id: String,
    #[serde(default)]
    pub executable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeJson {
    pub entries: Vec<TreeEntryJson>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignatureJson {
    pub name: String,
    pub email: String,
    pub timestamp_millis: i64,
    pub tz_offset: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitJson {
    pub parents: Vec<String>,
    #[serde(default)]
    pub predecessors: Vec<String>,
    /// jj's `Merge<TreeId>`. One term is a resolved tree; more than one is a
    /// conflict, which Ledger does not model and refuses with a reason.
    pub root_tree: Vec<String>,
    pub change_id: String,
    pub description: String,
    pub author: SignatureJson,
    pub committer: SignatureJson,
}

#[derive(Debug, Deserialize)]
struct IdResponse {
    id: String,
}

#[derive(Debug, Deserialize)]
struct WriteCommitResponse {
    id: String,
    commit: CommitJson,
}

/// A Ledger environment, addressed over HTTP.
#[derive(Debug, Clone)]
pub struct LedgerClient {
    base: String,
    token: Option<String>,
    agent: ureq::Agent,
}

impl LedgerClient {
    /// `base` is the service root, e.g. `http://127.0.0.1:8080`; `env` is
    /// `org/environment`.
    pub fn new(base: &str, env: &str, token: Option<String>) -> Self {
        let config = ureq::Agent::config_builder()
            .timeout_global(Some(TIMEOUT))
            .build();
        Self {
            base: format!("{}/v1/envs/{}/jj", base.trim_end_matches('/'), env),
            token,
            agent: config.into(),
        }
    }

    pub fn info(&self) -> ClientResult<BackendInfo> {
        self.get_json("/info")
    }

    // ── files and symlinks ──────────────────────────────────────────────────

    pub fn write_file(&self, contents: &[u8]) -> ClientResult<String> {
        let response: IdResponse = self.post_bytes("/files", contents)?;
        Ok(response.id)
    }

    pub fn read_file(&self, id: &str) -> ClientResult<Vec<u8>> {
        self.get_bytes(&format!("/files/{id}"))
    }

    pub fn write_symlink(&self, target: &str) -> ClientResult<String> {
        let response: IdResponse =
            self.post_json("/symlinks", &serde_json::json!({ "target": target }))?;
        Ok(response.id)
    }

    pub fn read_symlink(&self, id: &str) -> ClientResult<String> {
        #[derive(Deserialize)]
        struct Target {
            target: String,
        }
        let response: Target = self.get_json(&format!("/symlinks/{id}"))?;
        Ok(response.target)
    }

    // ── trees ───────────────────────────────────────────────────────────────

    pub fn write_tree(&self, tree: &TreeJson) -> ClientResult<String> {
        let response: IdResponse = self.post_json("/trees", tree)?;
        Ok(response.id)
    }

    pub fn read_tree(&self, id: &str) -> ClientResult<TreeJson> {
        self.get_json(&format!("/trees/{id}"))
    }

    // ── commits ─────────────────────────────────────────────────────────────

    pub fn write_commit(&self, commit: &CommitJson) -> ClientResult<(String, CommitJson)> {
        let response: WriteCommitResponse = self.post_json("/commits", commit)?;
        Ok((response.id, response.commit))
    }

    pub fn read_commit(&self, id: &str) -> ClientResult<CommitJson> {
        self.get_json(&format!("/commits/{id}"))
    }

    // ── transport ───────────────────────────────────────────────────────────

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base, path)
    }

    /// The bearer header, or nothing when running against a `--dev` Ledger.
    ///
    /// One credential shape, the same one every other Ledger client uses: this
    /// backend has no idea of who anyone is that the service does not already
    /// have.
    fn auth_header(&self) -> String {
        match &self.token {
            Some(token) => format!("Bearer {token}"),
            None => String::new(),
        }
    }

    fn get_bytes(&self, path: &str) -> ClientResult<Vec<u8>> {
        let mut request = self.agent.get(self.url(path));
        let header = self.auth_header();
        if !header.is_empty() {
            request = request.header("Authorization", header);
        }
        let mut response = request.call().map_err(convert)?;
        response
            .body_mut()
            .read_to_vec()
            .map_err(|error| ClientError::Malformed(error.to_string()))
    }

    fn get_json<T: for<'de> Deserialize<'de>>(&self, path: &str) -> ClientResult<T> {
        let bytes = self.get_bytes(path)?;
        serde_json::from_slice(&bytes).map_err(|error| ClientError::Malformed(error.to_string()))
    }

    fn post<T: for<'de> Deserialize<'de>>(
        &self,
        path: &str,
        content_type: &str,
        body: &[u8],
    ) -> ClientResult<T> {
        let mut request = self
            .agent
            .post(self.url(path))
            .header("Content-Type", content_type);
        let header = self.auth_header();
        if !header.is_empty() {
            request = request.header("Authorization", header);
        }
        let mut response = request.send(body).map_err(convert)?;
        let bytes = response
            .body_mut()
            .read_to_vec()
            .map_err(|error| ClientError::Malformed(error.to_string()))?;
        serde_json::from_slice(&bytes).map_err(|error| ClientError::Malformed(error.to_string()))
    }

    fn post_json<B: Serialize, T: for<'de> Deserialize<'de>>(
        &self,
        path: &str,
        body: &B,
    ) -> ClientResult<T> {
        let payload =
            serde_json::to_vec(body).map_err(|error| ClientError::Malformed(error.to_string()))?;
        self.post(path, "application/json", &payload)
    }

    fn post_bytes<T: for<'de> Deserialize<'de>>(
        &self,
        path: &str,
        body: &[u8],
    ) -> ClientResult<T> {
        self.post(path, "application/octet-stream", body)
    }
}

/// Ledger's error contract carries state — a 404 is not the same as a 403, and
/// jj branches on the difference — so the status is kept rather than flattened.
fn convert(error: ureq::Error) -> ClientError {
    match error {
        ureq::Error::StatusCode(status) => ClientError::Status {
            status,
            message: String::from("see the response body"),
        },
        other => ClientError::Unreachable(other.to_string()),
    }
}
