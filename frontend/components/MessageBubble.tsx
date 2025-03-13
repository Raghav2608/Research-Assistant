"use client";
import Message, { Sender } from "@/types/Message";

export interface MessageBubbleProps {
  msg: Message;
}

export default function Chatlog({ msg }: MessageBubbleProps) {
  return (
    <div
      className={`w-full mb-12 flex ${
        msg.sender == Sender.User ? "justify-end" : "justify-start"
      }`}
    >
      <div
        className={`w-2/5 ${
          msg.sender == Sender.User ? "bg-primary" : "bg-info"
        } text-2xl p-3 rounded-xl text-white`}
      >
        {msg.message}
      </div>
    </div>
  );
}
