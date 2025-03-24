"use client";

import Chatbox from "@/components/Chatbox";
import Chatlog from "@/components/Chatlog";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Message from "@/types/Message";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const router = useRouter();
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL;
  // On mount, check if user is already authenticated
  useEffect(() => {
    const whoamiurl = `${backendUrl}/whoami`;
    // Call the /whoami endpoint which will return the user info if authenticated
    fetch(whoamiurl, {
      method: "GET",
      credentials: "include", // Send cookies
    })
      .then((res) => {
        if (!res.ok) {
          // If authenticated, redirect to home page
          router.push("/login");
        }
      })
      .catch((error) => console.error("Error checking auth status:", error));
  }, [router, backendUrl]);

  function addMessage(msg: Message): void {
    setMessages((prevMessages) => [...prevMessages, msg]);
  }
  return (
    <div className="flex flex-col-reverse items-center w-full">
      <Chatbox addMessage={addMessage} setIsLoading={setIsLoading}/>
      <Chatlog messages={messages} isLoading={isLoading}/>
    </div>
  );
}
