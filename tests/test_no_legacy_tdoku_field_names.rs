//! Regression guard: the legacy schema field names `tdoku_backtracks` and
//! `tdoku_unique` were renamed to `search_backtracks` / `search_unique` (iter-5
//! canonical naming — the values are produced by the in-process Rust
//! backtracker, not the external tdoku binary).
//!
//! This test greps the entire `src/` tree and asserts that the legacy names
//! never reappear in *write* paths or the canonical schema. Mentions of the
//! deprecated `--tdoku-bin` CLI flag (which refers to the *binary*, not the
//! field) are explicitly allowed. The `src/rerate.rs` reader is allowed to
//! reference legacy names for **input backwards-compat fallback** (v6/v7
//! sealed Python parquet shards used `tdoku_*`); output is always `search_*`.

#[test]
fn test_no_legacy_tdoku_field_names() {
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut offenders: Vec<String> = Vec::new();
    walk(&src, &mut |path, contents| {
        // Backwards-compat read fallback in rerate.rs is intentionally
        // allowed to mention the legacy column names.
        let is_rerate = path.file_name().and_then(|s| s.to_str()) == Some("rerate.rs");
        for (i, line) in contents.lines().enumerate() {
            if line.contains("tdoku_backtracks") || line.contains("tdoku_unique") {
                if is_rerate {
                    continue;
                }
                offenders.push(format!("{}:{}: {}", path.display(), i + 1, line.trim()));
            }
        }
    });
    assert!(
        offenders.is_empty(),
        "legacy schema field name(s) re-introduced in src/:\n{}",
        offenders.join("\n"),
    );
}

fn walk(dir: &std::path::Path, f: &mut dyn FnMut(&std::path::Path, &str)) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for ent in entries.flatten() {
        let path = ent.path();
        if path.is_dir() {
            walk(&path, f);
        } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            if let Ok(contents) = std::fs::read_to_string(&path) {
                f(&path, &contents);
            }
        }
    }
}
