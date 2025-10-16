# Vercel Frontend Deployment Guide

## Prerequisites

1. **GitHub Repository**: Frontend code pushed to GitHub
2. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
3. **Backend API**: Deployed on Render (get the URL)

## Deployment Steps

### 1. Prepare Frontend

```bash
cd frontend

# Create production environment file
cp .env.example .env.production
# Edit .env.production with your Render API URL
```

### 2. Update Environment Variables

Edit `frontend/.env.production`:
```env
VITE_API_URL=https://your-render-app.onrender.com
```

### 3. Test Build Locally

```bash
cd frontend

# Install dependencies
npm install

# Test production build
npm run build

# Test preview
npm run preview
```

### 4. Deploy to Vercel

#### Option A: Vercel CLI (Recommended)

```bash
# Install Vercel CLI
npm install -g vercel

# Login to Vercel
vercel login

# Deploy from frontend directory
cd frontend
vercel

# Follow prompts:
# - Set up and deploy? Yes
# - Which scope? Your account
# - Link to existing project? No
# - Project name: log-anomaly-frontend
# - Directory: ./
# - Override settings? No

# Deploy to production
vercel --prod
```

#### Option B: Vercel Dashboard

1. **Login to Vercel Dashboard**
   - Go to [vercel.com/dashboard](https://vercel.com/dashboard)
   - Connect your GitHub account

2. **Import Project**
   - Click "New Project"
   - Import your GitHub repository
   - Select the `frontend` folder as root directory

3. **Configure Build Settings**
   - **Framework Preset**: Vite
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
   - **Install Command**: `npm install`

4. **Environment Variables**
   ```
   VITE_API_URL = https://your-render-app.onrender.com
   ```

5. **Deploy**
   - Click "Deploy"
   - Wait for deployment (2-5 minutes)

### 5. Configure Custom Domain (Optional)

1. **Go to Project Settings**
   - Select your project
   - Go to "Domains" tab

2. **Add Domain**
   - Enter your domain name
   - Follow DNS configuration instructions

3. **SSL Certificate**
   - Vercel automatically provides SSL
   - Certificate is issued within minutes

### 6. Test Deployment

```bash
# Test the deployed frontend
curl https://your-app.vercel.app

# Test API connectivity
# Open browser developer tools and check network requests
```

## Configuration Files

### vercel.json
```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "framework": "vite",
  "installCommand": "npm install",
  "devCommand": "npm run dev",
  "env": {
    "VITE_API_URL": "@api_url"
  },
  "build": {
    "env": {
      "VITE_API_URL": "@api_url"
    }
  },
  "rewrites": [
    {
      "source": "/api/(.*)",
      "destination": "$VITE_API_URL/api/$1"
    }
  ]
}
```

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `VITE_API_URL` | Backend API URL | `https://log-anomaly-api.onrender.com` |

## Troubleshooting

### Common Issues

1. **API Connection Errors**
   ```javascript
   // Check CORS settings in backend
   // Ensure API URL is correct
   console.log('API URL:', import.meta.env.VITE_API_URL);
   ```

2. **Build Failures**
   ```bash
   # Check for TypeScript errors
   npm run build

   # Check dependencies
   npm audit fix
   ```

3. **Environment Variables Not Working**
   ```bash
   # Ensure variables start with VITE_
   # Redeploy after changing environment variables
   vercel --prod
   ```

### Performance Optimization

1. **Bundle Analysis**
   ```bash
   # Install bundle analyzer
   npm install --save-dev rollup-plugin-visualizer

   # Add to vite.config.js
   import { visualizer } from 'rollup-plugin-visualizer';
   
   export default defineConfig({
     plugins: [
       react(),
       tailwindcss(),
       visualizer()
     ]
   });
   ```

2. **Code Splitting**
   ```javascript
   // Lazy load components
   const LazyComponent = lazy(() => import('./Component'));
   ```

3. **Image Optimization**
   ```javascript
   // Use Vercel Image Optimization
   import Image from 'next/image'; // If using Next.js
   ```

## Monitoring and Analytics

### 1. Vercel Analytics

```bash
# Install Vercel Analytics
npm install @vercel/analytics

# Add to main.jsx
import { Analytics } from '@vercel/analytics/react';

function App() {
  return (
    <>
      <YourApp />
      <Analytics />
    </>
  );
}
```

### 2. Performance Monitoring

```javascript
// Add performance monitoring
const observer = new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    console.log('Performance:', entry);
  }
});

observer.observe({ entryTypes: ['navigation', 'paint'] });
```

### 3. Error Tracking

```javascript
// Add error boundary
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    console.error('Error caught:', error, errorInfo);
    // Send to error tracking service
  }

  render() {
    if (this.state.hasError) {
      return <h1>Something went wrong.</h1>;
    }
    return this.props.children;
  }
}
```

## Security Considerations

1. **Environment Variables**
   - Never expose sensitive data in VITE_ variables
   - Use server-side environment variables for secrets

2. **Content Security Policy**
   ```json
   // In vercel.json
   {
     "headers": [
       {
         "source": "/(.*)",
         "headers": [
           {
             "key": "Content-Security-Policy",
             "value": "default-src 'self'; script-src 'self' 'unsafe-inline'"
           }
         ]
       }
     ]
   }
   ```

3. **HTTPS Enforcement**
   ```json
   // In vercel.json
   {
     "headers": [
       {
         "source": "/(.*)",
         "headers": [
           {
             "key": "Strict-Transport-Security",
             "value": "max-age=31536000; includeSubDomains"
           }
         ]
       }
     ]
   }
   ```

## Continuous Deployment

### Automatic Deployments

1. **GitHub Integration**
   - Vercel automatically deploys on push to main branch
   - Preview deployments for pull requests

2. **Branch Deployments**
   ```json
   // In vercel.json
   {
     "git": {
       "deploymentEnabled": {
         "main": true,
         "develop": true
       }
     }
   }
   ```

### Deployment Hooks

```bash
# Add deployment webhook
curl -X POST https://api.vercel.com/v1/integrations/deploy/your-hook-id
```

## Advanced Configuration

### 1. Redirects and Rewrites

```json
// In vercel.json
{
  "redirects": [
    {
      "source": "/old-path",
      "destination": "/new-path",
      "permanent": true
    }
  ],
  "rewrites": [
    {
      "source": "/api/(.*)",
      "destination": "https://your-api.onrender.com/api/$1"
    }
  ]
}
```

### 2. Edge Functions

```javascript
// api/hello.js
export default function handler(request) {
  return new Response(`Hello from ${request.geo.city}!`);
}
```

### 3. Middleware

```javascript
// middleware.js
import { NextResponse } from 'next/server';

export function middleware(request) {
  // Add custom headers
  const response = NextResponse.next();
  response.headers.set('X-Custom-Header', 'value');
  return response;
}
```

Your frontend will be available at: `https://your-app.vercel.app`

## Complete Deployment Checklist

- [ ] Backend deployed on Render
- [ ] Frontend environment variables configured
- [ ] Build tested locally
- [ ] Deployed to Vercel
- [ ] API connectivity verified
- [ ] Custom domain configured (optional)
- [ ] Analytics and monitoring set up
- [ ] Error tracking implemented
- [ ] Performance optimized