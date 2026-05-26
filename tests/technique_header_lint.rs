//! Enforce header-doc convention on all `techniques/*.rs` files.
//! AlphaEvolve / OpenEvolve relies on this contract.

use std::fs;
use std::path::Path;

const REQUIRED_SECTIONS: &[&str] = &[
    "## Inputs",
    "## Mutates",
    "## Returns",
    "## Performance budget",
    "## Algorithm reference",
    "## AlphaEvolve contract",
];

#[test]
fn all_technique_files_have_header_doc() {
    let dir = Path::new("src/generic/techniques");
    let mut missing = Vec::new();
    for entry in fs::read_dir(dir).expect("read techniques dir") {
        let path = entry.unwrap().path();
        if path.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        if name == "mod.rs" || name == "result.rs" {
            continue;
        }
        let body = fs::read_to_string(&path).unwrap();
        let head: String = body.lines().take(40).collect::<Vec<_>>().join("\n");
        for sec in REQUIRED_SECTIONS {
            if !head.contains(sec) {
                missing.push(format!("{name}: missing `{sec}`"));
            }
        }
    }
    assert!(missing.is_empty(), "header-doc violations:\n{}", missing.join("\n"));
}
