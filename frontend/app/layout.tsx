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
        <header className="flex flex-row items-center justify-center w-full bg-dark h-14">
          <h1>
            <span className="text-center text-primary font-extrabold text-3xl">RESEARCH ASSISTANT</span>
          </h1>
        </header>
        <main className="flex flex-row flex-1 w-full bg-light">{children}</main>
        <footer className="flex flex-row items-center justify-center w-full bg-dark h-10">
          <div>
            <span className="text-info">Devloped by group 4 for LGP</span>
          </div>
        </footer>
      </body>
    </html>
  );
}
