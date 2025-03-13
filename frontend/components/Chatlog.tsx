"use client";
import Message from "@/types/Message";

export interface ChatlogProps {
  messages: Message[];
}

export default function Chatlog({ messages }: ChatlogProps) {
  return (
    <div className="w-3/5 flex flex-col-reverse h-full my-10 overflow-y-auto"></div>
  );
}
