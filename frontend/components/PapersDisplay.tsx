"use client";
import React from "react";
import Paper from "@/types/Paper";

interface PapersDisplayProps {
  papers: Paper[];
}

export default function PapersDisplay({ papers }: PapersDisplayProps) {
  return (
    <div className="w-full flex flex-row flex-wrap gap-4 p-4">
      {papers.map((paper, index) => (
        <a
          key={index}
          href={paper.metadata.link}
          target="_blank"
          rel="noopener noreferrer"
          className="px-4 py-2 bg-lightest text-black font-bold rounded hover:bg-primary-dark transition-colors whitespace-nowrap"
        >
          {paper.metadata.title}{" "}
        </a>
      ))}
    </div>
  );
}
