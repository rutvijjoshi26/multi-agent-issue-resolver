CREATE TABLE IF NOT EXISTS issues (
    number       INTEGER PRIMARY KEY,
    title        TEXT NOT NULL,
    body         TEXT NOT NULL,
    labels       TEXT[] NOT NULL
);

CREATE TABLE IF NOT EXISTS pull_requests (
    number        INTEGER PRIMARY KEY,
    merged        BOOLEAN NOT NULL,
    title         TEXT NOT NULL,
    changed_files TEXT[] NOT NULL,
    base_commit   TEXT NOT NULL DEFAULT ''
);

-- Added after the table shipped; CREATE TABLE IF NOT EXISTS would skip it
-- on a database created by an earlier version.
ALTER TABLE pull_requests
    ADD COLUMN IF NOT EXISTS base_commit TEXT NOT NULL DEFAULT '';

CREATE TABLE IF NOT EXISTS issue_pr_links (
    issue_number INTEGER NOT NULL REFERENCES issues(number) ON DELETE CASCADE,
    pr_number    INTEGER NOT NULL REFERENCES pull_requests(number) ON DELETE CASCADE,
    PRIMARY KEY (issue_number, pr_number)
);
