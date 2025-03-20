import { useState } from "react";
import { useRouter } from "next/navigation";

export default function RegisterForm() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const router = useRouter();

  const handleRegister = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    console.log("Registering:", { username, password });
    // Add API request here
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-light">
      <form
        onSubmit={handleRegister}
        className="bg-white p-6 rounded-2xl shadow-md w-80"
      >
        <h2 className="text-xl font-bold mb-4">Register</h2>
        <input
          type="text"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          className="w-full p-2 border rounded mb-3 focus:outline-none focus:ring-2 focus:ring-primary"
          required
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="w-full p-2 border rounded mb-3 focus:outline-none focus:ring-2 focus:ring-primary"
          required
        />
        <button
          type="submit"
          className="w-full bg-primary text-white py-2 rounded hover:bg-secondary transition"
        >
          Register
        </button>
      </form>
      <button
        onClick={() => router.push("/login")}
        className="mt-4 text-primary underline hover:text-secondary"
      >
        Already have an account? Login
      </button>
    </div>
  );
}
