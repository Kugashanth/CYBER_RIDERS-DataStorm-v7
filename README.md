# CYBER_RIDERS — Data Storm v7.0 Final Round

> **Data Storm v7.0** is Sri Lanka's premier data science competition organized by the 
> Rotaract Club of University of Moratuwa, powered by OCTAVE (John Keells Group).

## 🧠 About This Project

This repository contains the end-to-end data science solution developed by team 
**CYBER_RIDERS** for the Final Round of Data Storm v7.0.

The challenge involves building an enterprise-grade **Outlet Sales Potential Prediction 
Engine** for a leading Sri Lankan beverage manufacturer operating across 80,000+ 
traditional retail outlets (kades, groceries, eateries, pharmacies) island-wide.

Rather than relying on historical sales averages, our solution estimates the **Maximum 
Monthly Purchase Potential** of 20,000 traditional trade outlets across 4 provinces 
(Western, Central, North-Western, Southern) for **January 2026** — shifting the business 
from reactive historical allocation to **Potential-Based Trade Marketing**.

---

## 🎯 Problem Statement

A beverage distributor allocates coolers, trade budgets, and promotional discounts based 
on what outlets *did* sell — not what they *could* sell. This creates two critical blind spots:

- **Underperforming high-potential outlets** (e.g., busy town-center kades limited by 
  poor stock or credit constraints) receive inadequate resources.
- **Maxed-out low-potential outlets** (e.g., small rural shops already at ceiling) 
  receive over-investment.

Our solution uncovers the **latent demand signal** hidden beneath censored historical 
data and translates it into actionable trade marketing decisions.

---

## 🔑 Key Features

- 🏗️ **Medallion Lakehouse Architecture** — Structured Bronze → Silver → Gold data 
  pipeline with idempotent processing and a dedicated Rejected Records Store
- 📍 **Spatial Distance-Decay Modeling** — Gravity model and Gaussian decay functions 
  applied to OpenStreetMap POI data for non-linear proximity scoring
- 🏪 **Competitive Catchment Density** — Spatial clustering to estimate market 
  saturation and competitor intensity around each outlet
- 📈 **Censored Demand Modeling** — Tobit regression and quantile regression to 
  uncap artificially suppressed historical sales and reveal true latent potential
- 💰 **LKR 5M Budget Optimizer** — Linear programming model to allocate the Western 
  Province promotional budget across distributors and outlets to maximize volume lift
- 🤖 **Explainable AI (XAI) Module** — SHAP-based feature importance combined with 
  LLM-generated plain-language business explanations for every outlet prediction
- 🌐 **Outlet Intelligence Web App** — Interactive dashboard to browse, filter, and 
  drill into outlet-level predictions with AI-generated reasoning narratives

---

## 📦 Deliverables

| # | Deliverable | File |
|---|---|---|
| 1 | Latent Potential Predictions | `CYBER_RIDERS_predictions.csv` |
| 2 | Marketing Budget Allocations | `CYBER_RIDERS_budget_allocations.csv` |
| 3 | Full Codebase | This repository |
| 4 | Outlet Intelligence Web App | `/webapp` |
| 5 | Technical Methodology Paper | `/docs/technical_paper.pdf` |
| 6 | Executive Pitch Deck | `/docs/pitch_deck.pdf` |

---

## 🗂️ Repository Structure
