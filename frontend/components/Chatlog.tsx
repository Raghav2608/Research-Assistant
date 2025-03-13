"use client";
import Message, { Sender } from "@/types/Message";
import MessageBubble from "./MessageBubble";
import { useEffect } from "react";

export interface ChatlogProps {
  messages: Message[];
}

export default function Chatlog({ messages }: ChatlogProps) {
  useEffect(() => {
    // Scroll to bottom when messages change
    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  }, [messages]);

  return (
    <div className="w-3/5 flex flex-col h-full mt-32 overflow-y-auto">
      {messages.map((msg, index) => {
        return <MessageBubble msg={msg} key={index} />;
      })}
    </div>
  );
}
