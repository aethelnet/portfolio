# Aethelburg Systems: Engineering Portfolio & Research Terminal

Production portfolio hub for autonomous neurosymbolics, continuous dynamical systems, C++20 real-time audio DSP, and cyber-physical networks.

## Architecture

- **Continuous Dynamic Canvas**: 60 FPS phase-space trajectory integration via 4th-Order Runge-Kutta (RK4) ODE solver.
- **Framework**: Astro 5 (Static Site Generation), Tailwind CSS v4, TypeScript.
- **Design System**: Dutch White Light Brutalism with responsive dark/light adaptive CSS variable architecture.
- **Metadata**: Schema.org JSON-LD `ProfilePage` structured data, OpenGraph, and Twitter card integrations.
- **Host**: Reverse-proxied via Nginx on `server-141` (`https://aethelburg.info`).

## Directory Structure

```text
apps/portfolio/
├── public/
│   ├── favicon.ico
│   ├── favicon.svg          # Vector architectural [Æ] monogram
│   └── og-preview.svg       # OpenGraph card preview
├── src/
│   ├── components/          # Interactive dossiers and modal laboratories
│   │   ├── ArcScrubberModal.astro
│   │   ├── AudioEngineModal.astro
│   │   ├── EdisonReplayModal.astro
│   │   ├── LgnnSandboxModal.astro
│   │   ├── MuseumGraphModal.astro
│   │   ├── PhaseSpaceNavigator.astro
│   │   ├── ProjectDossier.astro
│   │   ├── ProjectIcon.astro
│   │   └── VoltGridModal.astro
│   ├── pages/
│   │   └── index.astro      # Main editorial layout & stage
│   └── styles/
│       └── global.css       # Brutalist design tokens & theme layer
└── package.json
```

## Local Development & Build

```bash
# Install dependencies
npm install

# Start local development server
npm run dev

# Compile production bundle
npm run build

# Preview static distribution
npm run preview
```

## Production Deployment

The static distribution `./dist/` is deployed to the production web server:

```bash
rsync -avz apps/portfolio/dist/ server-141:/home/ubuntu/portfolio/dist/
```
