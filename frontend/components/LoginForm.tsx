import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

export default function LoginForm() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const router = useRouter();
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL;

  // On mount, check if user is already authenticated
  useEffect(() => {
    const whoamiurl = `${backendUrl}/whoami`;
    fetch(whoamiurl, {
      method: "GET",
      credentials: "include", // Send cookies
    })
      .then((res) => {
        if (res.ok) {
          router.push("/");
        }
      })
      .catch((error) => console.error("Error checking auth status:", error));
  }, [router, backendUrl]);

  const handleLogin = async (
    e: React.FormEvent<HTMLFormElement> | React.KeyboardEvent<HTMLInputElement>
  ) => {
    e.preventDefault();
    console.log("Logging in:", { username, password });
    if (!backendUrl) {
      console.error("Backend URL is not defined in environment variables");
      return;
    }
    const loginUrl = `${backendUrl}/user_authentication`;

    try {
      const res = await fetch(loginUrl, {
        method: "POST",
        credentials: "include",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ username, password }),
      });

      if (res.ok) {
        router.push("/");
      } else {
        const data = await res.json();
        console.error("Login error:", data.message);
        alert("Login failed: " + data.message);
      }
    } catch (error) {
      console.error("Error logging in:", error);
    }
  };

  // Optional: onKeyDown handler to explicitly trigger login on Enter key press.
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleLogin(e);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-light py-12 px-4 sm:px-6 lg:px-8">
      <form
        onSubmit={handleLogin}
        className="bg-lighter p-8 rounded-2xl shadow-lg w-full max-w-md"
      >
        <h2 className="text-3xl font-bold mb-6 text-primary text-center">
          Login
        </h2>
        <input
          type="text"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          onKeyDown={handleKeyDown}
          className="w-full py-3 px-4 border bg-light rounded mb-4 focus:outline-none focus:ring-2 focus:ring-primary text-white"
          required
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          onKeyDown={handleKeyDown}
          className="w-full py-3 px-4 border bg-light rounded mb-4 focus:outline-none focus:ring-2 focus:ring-primary text-primary"
          required
        />
        <button
          type="submit"
          className="w-full bg-primary text-white py-3 rounded hover:bg-secondary transition"
        >
          Login
        </button>
      </form>
      <button
        onClick={() => router.push("/register")}
        className="mt-6 text-primary underline hover:text-secondary text-lg"
      >
        First time visiting the page? Register
      </button>
    </div>
  );
}
