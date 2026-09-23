# Shared Contracts

Contains shared runtime storage and contracts used by the bot and control plane.

The bot writes sanitized Binance Spot balance snapshots to the runtime SQLite database. The API reads those snapshots without receiving Binance credentials; exchange balances remain separate from the manually recorded Treasury ledger.
