"use client";
import Message, { Sender } from "@/types/Message";
import MessageBubble from "./MessageBubble";
import { useEffect } from "react";
import PapersDisplay from "./PapersDisplay";

export interface ChatlogProps {
  messages: Message[];
  isLoading: boolean;
}

const WELCOME_TEXT = `Hello and welcome! I'm your personal research assistant, here to help you navigate the world of academic research with ease. Ask me anything about the latest studies, breakthroughs, or trends, and I'll provide you with a clear summary and the sources I used.

For example, you might ask:

"What are the latest updates in transformer models?"
"Can you summarize recent advancements in computer vision?"
"What are the emerging trends in renewable energy research?"

Feel free to ask about any academic topic you're curious about, and let's start learning together!`;

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
      {isLoading ? (
        <MessageBubble
          msg={{ message: "Answering your question...", sender: Sender.Bot }}
        />
      ) : (
        ""
      )}
    </div>
  );
}
