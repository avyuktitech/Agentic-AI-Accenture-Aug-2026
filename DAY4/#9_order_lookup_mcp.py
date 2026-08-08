# order_lookup_mcp.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("order-lookup")

@mcp.tool()
def lookup_order(order_id: str) -> str:
    """Look up an order status by ID."""
    fake_db = {"1001": "Shipped", "1002": "Processing", "1003": "Delivered"}
    return fake_db.get(order_id, "Order not found")

if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8000)