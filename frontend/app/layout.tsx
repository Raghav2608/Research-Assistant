import React from "react";
import "./global.css";

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
      <body className="flex flex-col min-h-screen">
        <header className="flex flex-row w-full bg-dark">
          <span>placeholder for logo</span>
        </header>
        <main className="flex flex-row flex-1 w-full bg-light">{children}</main>
        <footer className="flex flex-row w-full bg-dark ">
          <span>Devloped by group 4 for LGP</span>
        </footer>
      </body>
    </html>
  );
}
