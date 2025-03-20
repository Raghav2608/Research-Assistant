import { useState } from "react";
import { useRouter } from "next/navigation";

export default function LoginForm() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const router = useRouter();

  const handleLogin = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    console.log("Loggin In:", { username, password });
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-light py-12 px-4 sm:px-6 lg:px-8">
      <form
        onSubmit={handleLogin}
        className="bg-lighter p-8 rounded-2xl shadow-lg w-full max-w-md"
      >
        <h2 className="text-3xl font-bold mb-6 text-primary text-center">Login</h2>
        <input
          type="text"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          className="w-full py-3 px-4 border bg-light rounded mb-4 focus:outline-none focus:ring-2 focus:ring-primary text-white"
          required
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
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
