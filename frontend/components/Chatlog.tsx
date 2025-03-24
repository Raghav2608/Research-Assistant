"use client";
import Message, { Sender } from "@/types/Message";
import MessageBubble from "./MessageBubble";
import { useEffect } from "react";
import PapersDisplay from "./PapersDisplay";

export interface ChatlogProps {
  messages: Message[];
  isLoading: boolean;
}

const WELCOME_TEXT =
  'Hi there I\'m your personal research assitant. Please ask me anything about academic research. For example "What are the latest updates in the transformer model world?". In return I will provide a summary of what I found alongside the sources I used.';
export default function Chatlog({ messages, isLoading }: ChatlogProps) {
  useEffect(() => {
    // Scroll to bottom when messages change
    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  }, [messages]);

  return (
    <div className="w-3/5 flex flex-col h-full mt-32 overflow-y-auto">
      <MessageBubble msg={{ message: WELCOME_TEXT, sender: Sender.Bot }} />
      {messages.map((msg, index) => {
        return (
          <div key={index}>
            <MessageBubble msg={msg} />
            {msg.papers && msg.papers.length > 0 && (
              <PapersDisplay papers={msg.papers} />
            )}
          </div>
        );
      })}
      {isLoading
        ? <MessageBubble
        msg={{ message: "Answering your question...", sender: Sender.Bot }}/>
        : ""}
    </div>
  );
}
