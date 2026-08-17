//! Real `jj`, with Ledger as its storage.
//!
//! There is an argument worth restating, because it is what this crate
//! exists to honour:
//!
//! > *"Native for agents" is usually read as "build a familiar-looking CLI". The
//! > stronger answer is not to build a client at all: jj already separates its
//! > interface from its storage, so agents run **real** jj. Commits, trees and
//! > files are Ledger's objects, bookmarks are refs, and jj's operation log is
//! > Ledger's — server-side and authoritative.*
//!
//! So this is not a jj-like tool. It is an implementation of
//! [`jj_lib::backend::Backend`], and everything above it is jj itself.
//!
//! # What this crate deliberately does not do
//!
//! **It never computes an object name.** It posts a jj-shaped object to Ledger,
//! and Ledger encodes it canonically and returns the name. A second
//! implementation of that encoding — in another language, maintained separately
//! — is precisely the silent-divergence failure Ledger's frozen format exists to
//! prevent: two encoders that agree today and disagree after one careless edit
//! would fork the corpus, and nothing anywhere would report an error.
//!
//! The price is a round trip per written object. Against a Ledger on the same
//! machine that is nothing, and it buys one canonical encoder forever.
//!
//! **It does not model conflicts.** jj represents a conflicted tree as a
//! `Merge<TreeId>` with more than one term. Ledger has no such object — its
//! merge refuses an overlap rather than storing one — so writing a conflicted
//! commit fails with the reason. That is the stated conflict gap,
//! surfaced rather than papered over.

pub mod backend;
pub mod client;


pub use backend::LedgerBackend;
pub use client::LedgerClient;
