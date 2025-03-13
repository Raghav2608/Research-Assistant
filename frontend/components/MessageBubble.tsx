"use client";
import Message, { Sender } from "@/types/Message";

export interface MessageBubbleProps {
  msg: Message;
}

export default function Chatlog({ msg }: MessageBubbleProps) {
  return (
    <div className={`bg-${msg.sender == Sender.User ? "primary" : "info"}`}>
      {msg.message}
    </div>
  );
}
