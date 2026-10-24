# Security — Model Ops Engine

| Control | Detail |
|---------|--------|
| API key | Set `API_KEY` → require `X-API-Key` on `POST /predict` |
| CORS | `CORS_ORIGINS` (comma-separated); defaults to local Vite |
| Headers | `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy` |
| Secrets | `.env` gitignored; template in `.env.example` |
| Tracking | Local MLflow SQLite by default — not a shared prod store |

```bash
export API_KEY='replace-me'
model-ops-serve
curl -H "X-API-Key: replace-me" -H 'Content-Type: application/json' \
  -d '{"vehicle_count":28,"pedestrian_count":5,"avg_speed_kmh":18.5,"occupancy_ratio":0.72,"hour_of_day":8,"is_weekend":0,"rain_mm":2.0}' \
  http://127.0.0.1:8091/predict
```

Out of scope: OAuth, model signing, remote registry ACLs.
