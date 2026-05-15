"use client";

import { FormEvent, useEffect, useState } from "react";

import AuthGate from "../../components/AuthGate";
import { fetchApi } from "../../lib/api";
import { formatDateTimeEu } from "../../lib/datetime";

type PortfolioSummary = {
  total_value: number;
  invested_capital: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number | null;
  dry_powder: number;
  dry_powder_pct: number | null;
  largest_position: PortfolioHolding | null;
  holding_count: number;
};

type PortfolioHolding = {
  asset_symbol: string;
  quote_symbol: string;
  quantity: number;
  average_cost: number | null;
  invested_capital: number;
  market_price: number | null;
  market_value: number | null;
  unrealized_pnl: number | null;
  unrealized_pnl_pct: number | null;
  allocation_pct: number | null;
  is_dry_powder: boolean;
};

type PortfolioTransaction = {
  id: number;
  transaction_id: string;
  executed_at: string;
  tx_type: PortfolioTransactionType;
  asset_symbol: string;
  quote_symbol: string;
  quantity: number;
  price: number | null;
  fee_amount: number | null;
  fee_asset: string | null;
  source: string;
  status: string;
  note: string | null;
};

type PortfolioPayload = {
  path: string;
  summary: PortfolioSummary;
  holdings: PortfolioHolding[];
  transactions: PortfolioTransaction[];
  transaction_count: number;
  warnings: string[];
};

type CreatePortfolioTransactionResponse = {
  transaction: PortfolioTransaction;
  portfolio: Omit<PortfolioPayload, "warnings">;
  warnings: string[];
};

type PortfolioTransactionType = "buy" | "sell" | "deposit" | "withdraw" | "adjustment";

type TransactionFormState = {
  tx_type: PortfolioTransactionType;
  asset_symbol: string;
  quantity: string;
  price: string;
  quote_symbol: string;
  fee_amount: string;
  fee_asset: string;
  executed_at: string;
  note: string;
};

const EMPTY_FORM: TransactionFormState = {
  tx_type: "buy",
  asset_symbol: "",
  quantity: "",
  price: "",
  quote_symbol: "USDT",
  fee_amount: "",
  fee_asset: "USDT",
  executed_at: "",
  note: "",
};

function asNumber(value: string): number | null {
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }
  const numeric = Number(trimmed);
  return Number.isFinite(numeric) ? numeric : null;
}

function formatNumber(value: number | null | undefined, maximumFractionDigits = 2): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "Awaiting Price";
  }
  return new Intl.NumberFormat("en-US", {
    minimumFractionDigits: 0,
    maximumFractionDigits,
  }).format(value);
}

function formatCurrency(value: number | null | undefined, maximumFractionDigits = 2): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "Awaiting Price";
  }
  return `$${formatNumber(value, maximumFractionDigits)}`;
}

function formatSignedCurrency(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "Awaiting Price";
  }
  const sign = value > 0 ? "+" : value < 0 ? "-" : "";
  return `${sign}$${formatNumber(Math.abs(value), 2)}`;
}

function formatPercent(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "Pending";
  }
  return `${formatNumber(value, 2)}%`;
}

function signedToneClass(value: number | null | undefined, baseClass: string): string {
  if (typeof value !== "number" || !Number.isFinite(value) || value === 0) {
    return `${baseClass} value-neutral`;
  }
  return `${baseClass} ${value > 0 ? "value-positive" : "value-negative"}`;
}

function transactionLabel(type: PortfolioTransactionType): string {
  if (type === "buy") {
    return "Buy";
  }
  if (type === "sell") {
    return "Sell";
  }
  if (type === "deposit") {
    return "Deposit";
  }
  if (type === "withdraw") {
    return "Withdraw";
  }
  return "Adjustment";
}

function transactionToneClass(type: PortfolioTransactionType): string {
  if (type === "buy" || type === "deposit") {
    return "treasury-ledger-type treasury-ledger-type-positive";
  }
  if (type === "sell" || type === "withdraw") {
    return "treasury-ledger-type treasury-ledger-type-negative";
  }
  return "treasury-ledger-type";
}

function mergePortfolioPayload(payload: Omit<PortfolioPayload, "warnings">, warnings: string[]): PortfolioPayload {
  return {
    ...payload,
    warnings,
  };
}

export default function TreasuryPage(): JSX.Element {
  const [portfolio, setPortfolio] = useState<PortfolioPayload | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [saving, setSaving] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [form, setForm] = useState<TransactionFormState>(EMPTY_FORM);

  async function loadPortfolio(): Promise<void> {
    setError(null);
    try {
      const payload = await fetchApi<PortfolioPayload>("/api/portfolio");
      setPortfolio(payload);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Treasury is unavailable.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadPortfolio();
  }, []);

  async function submitTransaction(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const response = await fetchApi<CreatePortfolioTransactionResponse>("/api/portfolio/transactions", {
        method: "POST",
        body: {
          tx_type: form.tx_type,
          asset_symbol: form.asset_symbol,
          quantity: asNumber(form.quantity),
          price: asNumber(form.price),
          quote_symbol: form.quote_symbol,
          fee_amount: asNumber(form.fee_amount),
          fee_asset: form.fee_asset,
          executed_at: form.executed_at,
          note: form.note,
        },
      });
      setPortfolio(mergePortfolioPayload(response.portfolio, response.warnings));
      setForm(EMPTY_FORM);
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : "Could not add treasure.");
    } finally {
      setSaving(false);
    }
  }

  const summary = portfolio?.summary;
  const holdings = portfolio?.holdings ?? [];
  const transactions = portfolio?.transactions ?? [];
  const largestBag = summary?.largest_position;

  return (
    <AuthGate>
      <section className="panel page-shell treasury-page-shell">
        <p className="dialog-scrooge">Scrooge counts the vault before any coin gets a crown.</p>

        <section className="treasury-overview">
          <header className="treasury-section-head">
            <div>
              <h1>Treasury Overview</h1>
              <p className="muted">Manual spot ledger today, control-plane foundation tomorrow.</p>
            </div>
            <button type="button" className="dialog-user-btn treasury-refresh-btn" onClick={() => void loadPortfolio()} disabled={loading}>
              Refresh Treasury
            </button>
          </header>

          {error ? <p className="dialog-scrooge dialog-scrooge-error">{error}</p> : null}
          {loading ? <p className="status-performance-note">Scrooge counts the vault...</p> : null}

          <div className="treasury-summary-grid">
            <div className="treasury-summary-card treasury-summary-card-hero">
              <span className="treasury-summary-label">Total Treasure</span>
              <strong>{formatCurrency(summary?.total_value ?? 0)}</strong>
            </div>
            <div className="treasury-summary-card">
              <span className="treasury-summary-label">Invested Capital</span>
              <strong>{formatCurrency(summary?.invested_capital ?? 0)}</strong>
            </div>
            <div className="treasury-summary-card">
              <span className="treasury-summary-label">Floating Gain</span>
              <strong className={signedToneClass(summary?.unrealized_pnl, "treasury-summary-value")}>
                {formatSignedCurrency(summary?.unrealized_pnl ?? 0)}
              </strong>
              <span className="treasury-summary-note">{formatPercent(summary?.unrealized_pnl_pct)}</span>
            </div>
            <div className="treasury-summary-card">
              <span className="treasury-summary-label">Dry Powder</span>
              <strong>{formatCurrency(summary?.dry_powder ?? 0)}</strong>
              <span className="treasury-summary-note">{formatPercent(summary?.dry_powder_pct)} of vault</span>
            </div>
            <div className="treasury-summary-card">
              <span className="treasury-summary-label">Largest Bag</span>
              <strong>{largestBag ? largestBag.asset_symbol : "No Treasure Yet"}</strong>
              <span className="treasury-summary-note">{largestBag ? formatPercent(largestBag.allocation_pct) : "Awaiting first coin"}</span>
            </div>
          </div>
        </section>

        <section className="section-block treasury-form-panel">
          <header className="treasury-section-head">
            <div>
              <h2>Add Treasure</h2>
              <p className="muted">Each entry lands in the Treasury ledger and holdings are derived from it.</p>
            </div>
          </header>
          <form className="treasury-form" onSubmit={(event) => void submitTransaction(event)}>
            <label className="dialog-user-field">
              Action
              <select
                value={form.tx_type}
                onChange={(event) => setForm((current) => ({ ...current, tx_type: event.target.value as PortfolioTransactionType }))}
              >
                <option value="buy">Buy</option>
                <option value="sell">Sell</option>
                <option value="deposit">Deposit</option>
                <option value="withdraw">Withdraw</option>
                <option value="adjustment">Adjustment</option>
              </select>
            </label>
            <label className="dialog-user-field">
              Coin
              <input
                type="text"
                value={form.asset_symbol}
                placeholder="BTC"
                autoCapitalize="characters"
                onChange={(event) => setForm((current) => ({ ...current, asset_symbol: event.target.value.toUpperCase() }))}
                required
              />
            </label>
            <label className="dialog-user-field">
              Stack
              <input
                type="number"
                value={form.quantity}
                placeholder="0.25"
                min="0"
                step="any"
                onChange={(event) => setForm((current) => ({ ...current, quantity: event.target.value }))}
                required
              />
            </label>
            <label className="dialog-user-field">
              Entry Cost
              <input
                type="number"
                value={form.price}
                placeholder="65000"
                min="0"
                step="any"
                onChange={(event) => setForm((current) => ({ ...current, price: event.target.value }))}
              />
            </label>
            <label className="dialog-user-field">
              Quote
              <input
                type="text"
                value={form.quote_symbol}
                autoCapitalize="characters"
                onChange={(event) => {
                  const quote = event.target.value.toUpperCase();
                  setForm((current) => ({ ...current, quote_symbol: quote, fee_asset: current.fee_asset || quote }));
                }}
                required
              />
            </label>
            <label className="dialog-user-field">
              Fee
              <input
                type="number"
                value={form.fee_amount}
                placeholder="0"
                min="0"
                step="any"
                onChange={(event) => setForm((current) => ({ ...current, fee_amount: event.target.value }))}
              />
            </label>
            <label className="dialog-user-field">
              Fee Coin
              <input
                type="text"
                value={form.fee_asset}
                autoCapitalize="characters"
                onChange={(event) => setForm((current) => ({ ...current, fee_asset: event.target.value.toUpperCase() }))}
              />
            </label>
            <label className="dialog-user-field treasury-form-wide">
              Treasury Notes
              <input
                type="text"
                value={form.note}
                placeholder="Thesis, source, or exit thought"
                onChange={(event) => setForm((current) => ({ ...current, note: event.target.value }))}
              />
            </label>
            <button type="submit" className="dialog-user-btn treasury-submit-btn" disabled={saving}>
              {saving ? "Adding Treasure..." : "Add Treasure"}
            </button>
          </form>
        </section>

        <section className="section-block">
          <header className="treasury-section-head">
            <div>
              <h2>Vault Holdings</h2>
              <p className="muted">Current stacks derived from settled Treasury entries.</p>
            </div>
          </header>

          {holdings.length === 0 ? (
            <p className="trade-history-empty-sheet">The treasury is empty. Add your first treasure.</p>
          ) : (
            <div className="treasury-holdings-grid">
              {holdings.map((holding) => (
                <article key={`${holding.asset_symbol}-${holding.quote_symbol}`} className="treasury-holding-card">
                  <header className="treasury-holding-head">
                    <span className="treasury-coin">{holding.asset_symbol}</span>
                    <span className="treasury-share">{formatPercent(holding.allocation_pct)}</span>
                  </header>
                  <div className="treasury-holding-lines">
                    <div>
                      <span>Stack</span>
                      <strong>
                        {formatNumber(holding.quantity, 8)} {holding.asset_symbol}
                      </strong>
                    </div>
                    <div>
                      <span>Entry Cost</span>
                      <strong>{formatCurrency(holding.average_cost, 6)}</strong>
                    </div>
                    <div>
                      <span>Market Price</span>
                      <strong>{formatCurrency(holding.market_price, 6)}</strong>
                    </div>
                    <div>
                      <span>Treasure Value</span>
                      <strong>{formatCurrency(holding.market_value)}</strong>
                    </div>
                    <div>
                      <span>Floating Gain</span>
                      <strong className={signedToneClass(holding.unrealized_pnl, "treasury-inline-value")}>
                        {formatSignedCurrency(holding.unrealized_pnl)}
                      </strong>
                    </div>
                    <div>
                      <span>Share of Vault</span>
                      <strong>{formatPercent(holding.allocation_pct)}</strong>
                    </div>
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>

        <section className="section-block">
          <header className="treasury-section-head">
            <div>
              <h2>Treasury Ledger</h2>
              <p className="muted">Fresh manual entries arrive at the top.</p>
            </div>
          </header>

          {transactions.length === 0 ? (
            <p className="trade-history-empty-sheet">No Treasury entries yet.</p>
          ) : (
            <div className="treasury-ledger-stack">
              {transactions.map((transaction) => (
                <article key={transaction.transaction_id} className="treasury-ledger-row">
                  <span className={transactionToneClass(transaction.tx_type)}>{transactionLabel(transaction.tx_type)}</span>
                  <span className="treasury-ledger-main">
                    {formatNumber(transaction.quantity, 8)} {transaction.asset_symbol}
                    {transaction.price ? ` at ${formatCurrency(transaction.price, 6)}` : ""}
                  </span>
                  <span className="treasury-ledger-meta">{formatDateTimeEu(transaction.executed_at)}</span>
                </article>
              ))}
            </div>
          )}
        </section>

        {portfolio?.warnings.length ? (
          <section className="section-block">
            <h2>Red Flags</h2>
            <div className="dialog-scrooge dialog-scrooge-warning">
              {portfolio.warnings.map((warning) => (
                <p key={warning}>{warning}</p>
              ))}
            </div>
          </section>
        ) : null}
      </section>
    </AuthGate>
  );
}
