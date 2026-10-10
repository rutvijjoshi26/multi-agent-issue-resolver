from collections.abc import Sequence

import psycopg


def load_issue_text(
    conn: psycopg.Connection, numbers: Sequence[int]
) -> dict[int, tuple[str, str]]:
    """Title and body for each requested issue present in the corpus.

    The manifest pins case numbers and answers but not text, so the text
    is read from the same corpus the index was built from.
    """
    if not numbers:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT number, title, body FROM issues WHERE number = ANY(%s)",
            (list(numbers),),
        )
        return {number: (title, body) for number, title, body in cur.fetchall()}
