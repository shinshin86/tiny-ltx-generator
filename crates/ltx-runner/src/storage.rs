use anyhow::{Context, Result};
use serde::Serialize;
use std::{fs, path::Path};

pub fn ensure_dirs(paths: &[&str]) -> Result<()> {
    for path in paths {
        fs::create_dir_all(path).with_context(|| format!("create directory {path}"))?;
    }
    Ok(())
}

pub fn write_json<T: Serialize>(path: impl AsRef<Path>, value: &T) -> Result<()> {
    let raw = serde_json::to_string_pretty(value)?;
    fs::write(path.as_ref(), raw).with_context(|| format!("write {}", path.as_ref().display()))
}

pub fn append_jsonl<T: Serialize>(path: impl AsRef<Path>, value: &T) -> Result<()> {
    use std::io::Write;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path.as_ref())?;
    writeln!(file, "{}", serde_json::to_string(value)?)?;
    Ok(())
}

pub fn copy_dir_recursive(from: impl AsRef<Path>, to: impl AsRef<Path>) -> Result<()> {
    let from = from.as_ref();
    let to = to.as_ref();
    fs::create_dir_all(to).with_context(|| format!("create directory {}", to.display()))?;
    for entry in fs::read_dir(from).with_context(|| format!("read directory {}", from.display()))? {
        let entry = entry?;
        let src = entry.path();
        let dst = to.join(entry.file_name());
        if src.is_dir() {
            copy_dir_recursive(&src, &dst)?;
        } else {
            fs::copy(&src, &dst)
                .with_context(|| format!("copy {} to {}", src.display(), dst.display()))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn recursive_copy_preserves_nested_files() {
        let src = tempdir().unwrap();
        let dst = tempdir().unwrap();
        fs::create_dir_all(src.path().join("nested")).unwrap();
        fs::write(src.path().join("nested/file.txt"), "hello").unwrap();
        copy_dir_recursive(src.path(), dst.path().join("copy")).unwrap();
        let copied = fs::read_to_string(dst.path().join("copy/nested/file.txt")).unwrap();
        assert_eq!(copied, "hello");
    }
}

pub fn copy_dir_recursive(from: impl AsRef<Path>, to: impl AsRef<Path>) -> Result<()> {
    let from = from.as_ref();
    let to = to.as_ref();
    fs::create_dir_all(to).with_context(|| format!("create directory {}", to.display()))?;
    for entry in fs::read_dir(from).with_context(|| format!("read directory {}", from.display()))? {
        let entry = entry?;
        let src = entry.path();
        let dst = to.join(entry.file_name());
        if src.is_dir() {
            copy_dir_recursive(&src, &dst)?;
        } else {
            fs::copy(&src, &dst)
                .with_context(|| format!("copy {} to {}", src.display(), dst.display()))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn recursive_copy_preserves_nested_files() {
        let src = tempdir().unwrap();
        let dst = tempdir().unwrap();
        fs::create_dir_all(src.path().join("nested")).unwrap();
        fs::write(src.path().join("nested/file.txt"), "hello").unwrap();
        copy_dir_recursive(src.path(), dst.path().join("copy")).unwrap();
        let copied = fs::read_to_string(dst.path().join("copy/nested/file.txt")).unwrap();
        assert_eq!(copied, "hello");
    }
}
