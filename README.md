# Telltale — AI-Powered Misinformation & Content Assessment Platform

Telltale helps users assess suspicious online content by combining local machine-learning analysis with AI-generated, evidence-grounded explanations. It highlights potential misinformation signals, explains *why* a claim may be problematic, and shows the uncertainty behind each assessment.

## Live Demo

**Live URL:**https://amd-pink.vercel.app/

## The Problem

Misinformation spreads quickly, and most users see only a final verdict such as “fake” or “true” without understanding the reasoning behind it. This reduces trust and makes it difficult to act responsibly on online information.

Telltale addresses this by making AI assessments more transparent:

- It identifies potential misinformation signals.
- It explains the reasoning in plain language.
- It separates high-confidence findings from uncertain ones.
- It presents supporting evidence for review.

## Key Features

### 1. AI Result Explanation

After an assessment, Telltale sends structured analysis results to the AMD Token Factory hosted model. The model converts technical outputs into a clear, human-readable explanation.

Users can understand:

- Why the content received its assessment
- Which signals influenced the result
- What limitations or uncertainties exist

### 2. Multi-Model Analysis

Telltale supports multiple perspectives on the same content. Instead of relying on a single response, the platform can request analysis from more than one configured model through AMD Token Factory.

This helps compare:

- Different explanations
- Different levels of confidence
- Potential disagreements between models

### 3. Evidence & Uncertainty Review

Telltale does not present AI output as an unquestionable truth. It reviews the response for evidence quality and uncertainty.

The platform highlights:

- Claims that need additional verification
- Low-confidence or ambiguous findings
- Missing context
- Potential unsupported conclusions

## How AMD Token Factory Is Used

AMD Token Factory powers the core AI layer of Telltale through its hosted model API.

Our backend sends structured assessment evidence to the AMD-hosted model and receives:

- Plain-language explanations of ML results
- Multi-model analysis
- Evidence and uncertainty review

This makes AMD Token Factory an essential part of the working product, not an optional add-on.

```text
User submits content
        ↓
Local analysis generates structured signals
        ↓
Backend sends evidence to AMD Token Factory API
        ↓
AMD-hosted model returns explanation / analysis
        ↓
Telltale validates and displays the result
```

## Technology Stack

- **Frontend:** _Add: React / Next.js / Vite + React_
- **Backend:** _Add: FastAPI / Node.js / Next.js API routes_
- **Machine Learning:** LightGBM, SHAP, ONNX Runtime
- **Data Processing:** Python, pandas, NumPy, SciPy
- **AI Provider:** AMD Token Factory hosted model API
- **Deployment:** Vercel

## AMD Token Factory Models

Telltale uses the event-provided AMD Token Factory API for hosted inference.

**Model used:** _Add the exact model ID from your API response, for example: DeepSeek-V4-Flash_

The application uses AMD Token Factory for:

- AI Result Explanation
- Multi-Model Analysis
- Evidence & Uncertainty Review

## Setup

### 1. Clone the repository

```bash
git clone [https://github.com/HasnainAhmad67/AMD-DEVELOPERS-PROJECT.git](https://github.com/HasnainAhmad67/AMD-DEVELOPERS-PROJECT.git)
cd AMD-DEVELOPERS-PROJECT
```

### 2. Install dependencies

```bash
npm install
```

If the project uses Python:

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file:

```env
RADEON_API_KEY=your_token_factory_api_key
```

Do not commit the `.env` file or expose the API key in the frontend.

### 4. Run locally

```bash
npm run dev
```

## Deployment

The frontend is deployed on Vercel.

1. Import the GitHub repository into Vercel.
2. Add `RADEON_API_KEY` as a server-side environment variable.
3. Deploy the project.
4. Verify that AI-powered features work on the live URL.

## Screenshots

### Dashboard

_Add a screenshot here_

### AI Result Explanation

_Add a screenshot here_

### Multi-Model Analysis

_Add a screenshot here_

### Evidence & Uncertainty Review

_Add a screenshot here_

## Team

- **Hasnain Ahmad** — _Add your role, e.g. Full-Stack / AI Integration_
- _Add teammate name and role_

## Future Improvements

- Add source-link verification and citation checking
- Improve multimodal analysis for images and videos
- Add user feedback to improve explanation quality
- Expand uncertainty scoring across more content types
