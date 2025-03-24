"use client";
import Message, { Sender } from "@/types/Message";

export interface MessageBubbleProps {
  msg: Message;
}

// Helper function to parse the message string into HTML.
function parseMessageToHTML(message: string): string {
  // Replace **text** with <strong>text</strong>
  let html = message.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

  // Replace URLs with clickable, underlined links.
  // The regex captures:
  //    Group 1: the URL without trailing punctuation (no whitespace, ], or ) )
  //    Group 2 (optional): trailing punctuation characters.
  html = html.replace(/(https?:\/\/[^\s\]\)]+)([.,\]\)\};:!]+)?/g, (match, url, punctuation) => {
    // punctuation may be undefined; if so, use empty string.
    const trail = punctuation ? punctuation : "";
    return `<a href="${url}" target="_blank" rel="noopener noreferrer" style="text-decoration: underline;">${url}</a>${trail}`;
  });

  // Split the message into paragraphs using one or more blank lines.
  const paragraphs = html.split(/\n\s*\n/).map(p => p.trim()).filter(p => p.length > 0);

  // Wrap each paragraph in <p> tags.
  html = paragraphs.map(p => `<p>${p}</p>`).join("\n");
  return html;
}

export default function MessageBubble({ msg }: MessageBubbleProps) {
  return (
    <div
      className={`w-full mb-12 flex ${
        msg.sender == Sender.User ? "justify-end" : "justify-start"
      }`}
    >
      <div
        className={`max-w-5xl px-5 ${
          msg.sender == Sender.User ? "bg-primary" : "bg-info"
        } text-2xl p-3 rounded-xl text-white`}
        dangerouslySetInnerHTML={{ __html: parseMessageToHTML(msg.message) }}
      >
        
      </div>
    </div>
  );
}
