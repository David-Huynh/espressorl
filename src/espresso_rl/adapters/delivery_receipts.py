from espresso_rl.adapters.postgres_repositories import PostgresStore


class DatabaseDeliveryReceipts:
    """Adapter-owned replay journal, sharing the configured local database."""

    def __init__(self, store):
        self._store = store
        self._placeholder = "%s" if isinstance(store, PostgresStore) else "?"
        store.conn.execute("""CREATE TABLE IF NOT EXISTS mqtt_delivery_receipts (
            delivery_id TEXT PRIMARY KEY, outcome TEXT NOT NULL
        )""")
        store.conn.commit()

    def get(self, delivery_id):
        row = self._store.conn.execute(
            f"SELECT outcome FROM mqtt_delivery_receipts WHERE delivery_id={self._placeholder}",
            (delivery_id,),
        ).fetchone()
        return row["outcome"] if row else None

    def record(self, delivery_id, outcome):
        parameter = self._placeholder
        self._store.conn.execute(
            f"INSERT INTO mqtt_delivery_receipts(delivery_id,outcome) VALUES({parameter},{parameter}) "
            "ON CONFLICT(delivery_id) DO NOTHING", (delivery_id, outcome),
        )
        self._store.conn.commit()
