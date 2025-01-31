import React from "react";

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>My Research Assistant</title>
      </head>
      <body className="flex-col justify-between">
        <header className="flex-row w-full bg-dark"><span>placeholder for logo</span></header>
        <main className="flex-row w-full bg-light">{children}</main>
        <footer className="flex-row w-full bg-dark"><span>Devloped by group 4 for LGP</span></footer>
      </body>
    </html>
  );
}
