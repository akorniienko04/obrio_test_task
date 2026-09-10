# Sample App Store Review Analysis

## Application

- **App:** Nebula: Spiritual Guidance
- **App ID:** `1459969523`
- **Storefront:** United States (`us`)
- **Reviews analyzed:** 100
- **Source:** Apple App Store public reviews endpoint

## Rating metrics

| Rating | Count | Percentage |
|---:|---:|---:|
| 1★ | 45 | 45.0% |
| 2★ | 9 | 9.0% |
| 3★ | 6 | 6.0% |
| 4★ | 7 | 7.0% |
| 5★ | 33 | 33.0% |

**Average rating:** 2.74 / 5

## Sentiment

| Sentiment | Count | Percentage |
|---|---:|---:|
| Positive | 18 | 18.0% |
| Neutral | 7 | 7.0% |
| Negative | 75 | 75.0% |

## Common negative keywords

| Keyword | Mentions |
|---|---:|
| subscription | 66 |
| money | 59 |
| account | 36 |
| pay | 36 |
| free | 34 |
| charged | 33 |
| refund | 31 |
| scam | 29 |
| trial | 28 |

## Common negative phrases

| Phrase | Mentions |
|---|---:|
| free trial | 12 |
| cancel subscription | 9 |
| customer service | 6 |
| credit card | 6 |
| debit card | 5 |
| money back | 5 |
| pay subscription | 4 |
| day trial | 4 |

## Actionable insights

The recommendations below are based on the recurring negative payment and
subscription themes identified in the analyzed reviews.

1. **Make subscription cancellation easier.** The high frequency of
   `subscription` and `cancel subscription` indicates friction in subscription
   management. Make the cancellation path visible and keep it within the app.

2. **Clarify free-trial billing.** Reviews frequently mention `free trial`,
   `trial`, and `charged`. Show the exact trial end date, renewal price, and
   renewal date before confirmation and before the first charge.

3. **Improve refund and payment support.** `refund`, `money back`, `credit
   card`, and `debit card` indicate payment-related dissatisfaction. Provide a
   clear refund-status flow and a faster support path for disputed charges.

4. **Address trust concerns.** The repeated `scam` keyword suggests that
   unexpected billing or unclear trial terms are damaging user trust. Use
   explicit billing copy and proactive receipts to reduce confusion.

## Method and limitations

Reviews were randomly sampled from the available Apple storefront data and
deduplicated by review ID. Sentiment was classified by a multilingual
Transformer model. Keywords and phrases were counted only in reviews labeled
negative. Counts describe this 100-review sample and should not be interpreted
as the complete App Store population.
