# Méthodologie — Model Ops Engine

1. **Cadrage** — train → track (MLflow) → serve congestion score for Urban Pulse.  
2. **Réalisation** — pipeline `model-ops-train` + FastAPI `model-ops-serve`.  
3. **Vérification** — pytest API + métriques train.  
4. **Sécurité** — `SECURITY.md` (API key).  
5. **Livraison** — Dockerfile + GitHub Actions.
