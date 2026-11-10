import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'FacialPulse - Mental Wellness Screening',
  description: 'AI-powered facial behavior analysis for mental wellness screening',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet" />
        <link rel="icon" href="/logo.svg" type="image/svg+xml" />
      </head>
      <body style={{
        margin: 0,
        fontFamily: '"Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        background: 'linear-gradient(180deg, #f0fafa 0%, #e0f2f1 100%)',
        minHeight: '100vh'
      }}>
        {children}
      </body>
    </html>
  )
}
