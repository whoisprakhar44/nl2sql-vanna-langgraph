"""
create_sample_db.py — Generate a sample SQLite database for testing.

Run this once to create data/sample.db with realistic test data:

    python -m scripts.create_sample_db

The schema matches the YAML metadata files in data/metadata/.
"""

import sqlite3
import random
from datetime import datetime, timedelta
from pathlib import Path


DB_PATH = Path("./data/sample.db")

DDL = """
CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(150) UNIQUE NOT NULL,
    region VARCHAR(50) NOT NULL,
    created_at DATE NOT NULL
);

CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(200) NOT NULL,
    category VARCHAR(50) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    created_at DATE NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_date DATE NOT NULL,
    total DECIMAL(10,2) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS order_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity INTEGER NOT NULL DEFAULT 1,
    unit_price DECIMAL(10,2) NOT NULL
);
"""

REGIONS = ["North", "South", "East", "West"]
CATEGORIES = ["Electronics", "Clothing", "Home", "Books", "Food"]
STATUSES = ["pending", "shipped", "delivered", "cancelled"]
FIRST_NAMES = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank", "Ivy", "Jack"]
LAST_NAMES = ["Smith", "Johnson", "Lee", "Garcia", "Chen", "Patel", "Kim", "Brown", "Wilson", "Taylor"]
PRODUCT_NAMES = [
    "Wireless Headphones", "USB-C Hub", "Running Shoes", "Cotton T-Shirt",
    "Desk Lamp", "Coffee Maker", "Python Cookbook", "Organic Granola",
    "Smart Watch", "Yoga Mat", "Backpack", "Bluetooth Speaker",
    "Kitchen Scale", "Notebook Set", "Trail Mix", "Tablet Stand",
    "Wool Sweater", "Air Purifier", "Data Science Textbook", "Green Tea Pack",
]


def create_database() -> None:
    """Create the sample database with test data."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"Removed existing database: {DB_PATH}")

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Create tables.
    cursor.executescript(DDL)

    # -- Seed customers --------------------------------------------------------
    customers = []
    for i in range(1, 31):
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        name = f"{first} {last}"
        email = f"{first.lower()}.{last.lower()}{i}@example.com"
        region = random.choice(REGIONS)
        created = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 400))
        customers.append((name, email, region, created.strftime("%Y-%m-%d")))

    cursor.executemany(
        "INSERT INTO customers (name, email, region, created_at) VALUES (?, ?, ?, ?)",
        customers,
    )

    # -- Seed products ---------------------------------------------------------
    products = []
    for i, pname in enumerate(PRODUCT_NAMES):
        cat = CATEGORIES[i % len(CATEGORIES)]
        price = round(random.uniform(9.99, 299.99), 2)
        created = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 200))
        products.append((pname, cat, price, created.strftime("%Y-%m-%d")))

    cursor.executemany(
        "INSERT INTO products (name, category, price, created_at) VALUES (?, ?, ?, ?)",
        products,
    )

    # -- Seed orders + order_items ---------------------------------------------
    order_id = 0
    for _ in range(100):
        cust_id = random.randint(1, 30)
        order_date = datetime(2024, 6, 1) + timedelta(days=random.randint(0, 365))
        status = random.choices(STATUSES, weights=[15, 25, 45, 15])[0]

        # Build order items first to calculate total.
        num_items = random.randint(1, 4)
        items = []
        total = 0.0
        for _ in range(num_items):
            prod_id = random.randint(1, len(PRODUCT_NAMES))
            qty = random.randint(1, 5)
            unit_price = round(random.uniform(9.99, 199.99), 2)
            items.append((prod_id, qty, unit_price))
            total += qty * unit_price

        total = round(total, 2)
        cursor.execute(
            "INSERT INTO orders (customer_id, order_date, total, status) VALUES (?, ?, ?, ?)",
            (cust_id, order_date.strftime("%Y-%m-%d"), total, status),
        )
        order_id = cursor.lastrowid

        for prod_id, qty, unit_price in items:
            cursor.execute(
                "INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (?, ?, ?, ?)",
                (order_id, prod_id, qty, unit_price),
            )

    conn.commit()
    conn.close()

    print(f"Created sample database: {DB_PATH}")
    print(f"  - 30 customers")
    print(f"  - {len(PRODUCT_NAMES)} products")
    print(f"  - 100 orders")


if __name__ == "__main__":
    create_database()
