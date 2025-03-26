import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";

export default function RegisterForm() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const router = useRouter();
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL;

  // On mount, check if user is already authenticated
  useEffect(() => {
    const whoamiurl = `${backendUrl}/whoami`;
    fetch(whoamiurl, {
      method: "GET",
      credentials: "include",
    })
      .then((res) => {
        if (res.ok) {
          router.push("/");
        }
      })
      .catch((error) => console.error("Error checking auth status:", error));
  }, [router, backendUrl]);

  const handleRegister = async (
    e: React.FormEvent<HTMLFormElement> | React.KeyboardEvent<HTMLInputElement>
  ) => {
    e.preventDefault();
    console.log("Registering:", { username, password });

    if (!backendUrl) {
      console.error("Backend URL is not defined in environment variables");
      return;
    }

    const registerUrl = `${backendUrl}/user_authentication`;

    try {
      const res = await fetch(registerUrl, {
        method: "POST",
        credentials: "include",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          username,
          password,
          confirm_password: password,
        }),
      });

      if (res.ok) {
        router.push("/");
      } else {
        const data = await res.json();
        console.error("Register error:", data.message);
        alert("Register failed: " + data.message);
      }
    } catch (error) {
      console.error("Error during registration:", error);
    }
  };

  // onKeyDown handler to trigger registration on Enter
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleRegister(e);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-light py-12 px-4 sm:px-6 lg:px-8">
      <form
        onSubmit={handleRegister}
        className="bg-lighter p-8 rounded-2xl shadow-lg w-full max-w-md"
      >
        <h2 className="text-3xl font-bold mb-6 text-primary text-center">
          Register
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
          Register
        </button>
      </form>
      <button
        onClick={() => router.push("/login")}
        className="mt-6 text-primary underline hover:text-secondary text-lg"
      >
        Already have an account? Login
      </button>
    </div>
  );
}
