"use client";
import React from "react";

interface ModeButtonProps {
  mode: string;
  setMode: (mode: string) => void;
}

export default function ModeButton({ mode, setMode }: ModeButtonProps) {
  const toggleMode = () => {
    setMode(mode === "specific" ? "fast" : "specific");
  };

  return (
    <button
      onClick={toggleMode}
      className={`px-4 h-3/4 rounded border transition-colors flex items-center justify-center
        ${
          mode === "specific"
            ? "bg-primary text-white border-primary"
            : "bg-transparent border-info text-info hover:bg-info/10"
        }`}
    >
      <span className="font-bold text-lg">
        {mode === "specific" ? "Deep Mode: ON" : "Deep Mode: OFF"}{" "}
      </span>
    </button>
  );
}
