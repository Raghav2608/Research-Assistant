"use client";
import React from "react";

interface ModeButtonProps {
  mode: string;
  setMode: (mode: string) => void;
}

export default function ModeButton({ mode, setMode }: ModeButtonProps) {
  const toggleMode = () => {
    const newMode = mode === "fast" ? "specific" : "fast";
    setMode(newMode);
  };

  return (
    <button
      onClick={toggleMode}
      className="px-4 py-2 bg-secondary text-white rounded hover:bg-secondary-dark transition-colors"
    >
      {mode === "fast" ? "Switch to Deep" : "Switch to Fast"}
    </button>
  );
}