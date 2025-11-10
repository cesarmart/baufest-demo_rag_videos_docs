import './globals.css'

export const metadata = {
  title: 'Demo de Recuperación Inteligente de Información',
  description: 'Aplicación de consulta de PDF con chat multi-turn',
}

export default function RootLayout({ children }) {
  return (
    <html lang="es">
      <body>{children}</body>
    </html>
  )
}
