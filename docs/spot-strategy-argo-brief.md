# Scrooge Spot: короткий опис для Арго

Дата зрізу: 25 вересня 2026 року.

## Коротко

Scrooge керує наявним Spot-портфелем через незалежні цикли **Bargain**. Він не намагається вгадати абсолютний максимум або мінімум. Після достатньо сильного 24-годинного зростання ціни Scrooge продає частину позиції, а потім закриває саме цей Bargain зворотною купівлею після сприятливого руху щонайменше на 5% від його власної середньої ціни відкриття.

Є дві цілі:

- **Accumulate Cash**: продати частину активу на зростанні, викупити її дешевше та залишити прибуток у спільному `Vault Reserve`.
- **Accumulate Asset**: продати частину активу на зростанні та використати виручку, щоб викупити більше монет. Позитивний чистий приріст після повного закриття створює одноразовий idempotent Target-ratchet.

Нові Bargains відкриваються лише як **SELL-origin**. BUY opportunity використовується для їхнього прибуткового закриття та cleanup, але не створює новий BUY-origin exposure. Це правило увімкнене за замовчуванням і не має UI-перемикача.

`Minimum Holding` утворює захищену підлогу. Стратегія може продавати тільки частину Target вище цієї підлоги, яка фактично доступна в Binance custody. Cold Storage враховується у вартості портфеля, але не продається.

## Сигнал і розмір угоди

1. Щогодини порівнюється поточна ціна з ціною рівно 24 години тому.
2. Рівні абсолютного руху: **5% / 8% / 12% / 18%**.
3. Базові транші: **10% / 20% / 30% / 40%** доступної стратегічної місткості.
4. Зростання створює SELL opportunity, падіння створює BUY opportunity.
5. RSI, Bollinger Bands та EMA не змінюють напрямок, а лише масштабують транш: **0.5x / 1.0x / 1.25x / 1.5x**. ATR використовується тільки як контекст волатильності.
6. У межах одного directional campaign кожен новий рівень виконується не більше одного разу.
7. Закриття існуючого Bargain має пріоритет над відкриттям нового. Окрім звичайного profit close, waiter cleanup може закрити стару позицію за контрольованими правилами ризику. За один цикл дозволена максимум одна дія на актив.

Binance executor є авторитетним для `stepSize`, `tickSize`, `minQty`, `minNotional` та інших exchange filters. Облік використовує фактичні fills і зберігає комісію в її реальному asset без штучної конвертації.

## Умови обох backtests

- Реальні кількості, entry prices, custody та policy десяти активів на момент зрізу.
- Дев'ять активів торгуються; DOT повністю захищений у Cold Storage.
- Початковий вільний USDT: **$0**. Зворотні BUY для закриття SELL-origin Bargains використовують USDT, який перед цим згенерували simulated SELL.
- Binance Spot candles, інтервал 1h, 60 warm-up candles.
- Рішення після закриття candle N; fill за open candle N+1.
- Комісія: **0.1%**, slippage: **0 bps**.
- Open Bargains наприкінці не закриваються примусово, а mark-to-market.
- Для перейменованого активу використана зшита історія до та після міграції ринку з 53-годинною неторговою паузою.

## Портфельні результати

| Метрика | 6 місяців | 1 рік |
|---|---:|---:|
| Початкова ринкова вартість | $12,916.70 | $35,670.40 |
| Final Treasury Value | $16,433.63 | $17,394.53 |
| Результат Scrooge | +27.23% | -51.24% |
| Результат HODL | +30.08% | -52.90% |
| Різниця проти HODL | **-$368.77** | **+$592.13** |
| Maximum drawdown | -33.94% | -68.87% |
| Final Vault Reserve | $1,360.54 | $2,009.88 |
| Execution fees | $38.94 | $92.46 |

### A/B: вплив вимкнення BUY-origin

Обидві сторони A/B перераховані тим самим поточним engine та відрізняються лише дозволом на створення BUY-origin Bargains.

| Горизонт | BUY + SELL origins | Лише SELL-origin | Зміна Final Treasury |
|---|---:|---:|---:|
| 6 місяців | $16,405.13 | $16,433.63 | **+$28.50** |
| 1 рік | $16,835.87 | $17,394.53 | **+$558.66** |

Інтерактивні звіти:

- [Six-Month SELL-origin Replay](../runtime/spot_backtests/comparisons/20260925T-campaign-hold-fix/6m/sell-origin-only/report.html)
- [One-Year SELL-origin Replay](../runtime/spot_backtests/comparisons/20260925T-campaign-hold-fix/1y/sell-origin-only/report.html)
- [Six-Month BUY+SELL Baseline](../runtime/spot_backtests/comparisons/20260925T-campaign-hold-fix/6m/buy-and-sell-current/report.html)
- [One-Year BUY+SELL Baseline](../runtime/spot_backtests/comparisons/20260925T-campaign-hold-fix/1y/buy-and-sell-current/report.html)

Початкова вартість відрізняється між періодами, бо це ринкова оцінка тих самих початкових кількостей на дату старту конкретного replay, а не сума історичних внесків власника.

## Результати Bargains

| Метрика | 6 місяців | 1 рік |
|---|---:|---:|
| Відкрито Bargains | 311 | 671 |
| Повністю закрито | 285 | 645 |
| Closure rate | 91.64% | 96.13% |
| Залишилось open | 26 | 26 |
| Realized PnL, включно з cleanup | -$84.39 | +$986.12 |
| Unrealized open PnL | -$308.98 | -$311.60 |
| Bargain lifecycle PnL | **-$393.38** | **+$674.52** |
| Underwater open Bargains | 21 | 21 |
| Open 90+ днів | 0 | 0 |
| Median closed duration | 70.0 год | 65.0 год |
| P90 closed duration | 26.1 дня | 20.8 дня |

Closed PnL включає як profit closes, так і контрольовані збитки waiter cleanup, тому closure rate і realized PnL треба читати разом. Головні показники: результат проти HODL, lifecycle PnL, вік open Bargains та зайнятий ними інвентар.

## Що видно з результатів

1. **Вимкнення BUY-origin покращило обидва replay.** Ефект невеликий за 6 місяців (+$28.50), але суттєвий за рік (+$558.66), де стратегія тепер випереджає HODL на $592.13.
2. **Найбільша різниця проявилась у cleanup losses.** У річному A/B вони зменшилися з -$3,419.47 до -$1,415.74; maximum drawdown покращився з -70.77% до -68.87%.
3. **SELL-only не гарантує перевагу на кожному горизонті.** За 6 місяців стратегія все ще відстає від HODL на $368.77, а її Bargain lifecycle PnL становить -$393.38.
4. **Річний lifecycle став позитивним.** +$986.12 realized після cleanup та -$311.60 unrealized дають +$674.52; наприкінці немає Bargains старше 90 днів.
5. **Vault Reserve більше не витрачається на нові BUY-origin цикли.** У річному replay він завершує на $2,009.88 замість $546.83 у BUY+SELL baseline і залишається доступним для закриття SELL-origin Bargains.

## Обмеження тесту

- Це candle replay, а не симуляція order book, latency або market impact.
- Slippage у цих прогонах дорівнює нулю, тому execution assumptions оптимістичні.
- Використовуються поточні Binance filters, а не їхні історичні версії.
- Перевірено лише один фактичний market path для кожного горизонту.
- Відсутність початкового USDT може обмежувати ранні зворотні BUY для закриття SELL-origin Bargains.
- Open Bargains не мають примусового settlement наприкінці replay; натомість waiter cleanup застосовує age-, loss- і capacity-aware правила протягом replay.

## Питання для Арго

1. Чи варто колись повертати BUY-origin лише під суворим regime filter, чи залишити SELL-only як постійне правило?
2. Чи достатньо поточних waiter cleanup правил, чи потрібна окрема політика для Bargains старше 30/60/90 днів?
3. Чи потрібні ліміти на кількість одночасних Bargains, загальний open notional та exposure одного активу?
4. Чи має profit target залишатися фіксованим 5%, чи бути volatility/fee-aware?
5. Чи не надто активний Level 1 у 5%, і чи потрібен cooldown або campaign reset за іншими правилами?
6. Як коректно закривати underwater цикл: фіксований loss budget, portfolio-level netting, inventory rebalance чи заборона loss realization?
7. Які метрики треба оптимізувати першими: edge vs HODL, lifecycle PnL, drawdown, reserve growth, asset accumulation або capital lock duration?

Головна теза для обговорення: **SELL-origin-only прибирає збиткове джерело нового exposure, помітно покращує річний результат і drawdown, але шестимісячне відставання від HODL показує, що sizing, timing і cleanup ще потребують оптимізації.**
