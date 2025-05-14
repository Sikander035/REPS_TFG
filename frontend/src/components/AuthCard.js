import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

const AuthCard = () => {
    const [isSignUp, setIsSignUp] = useState(false);
    const navigate = useNavigate();

    const toggleForm = () => {
        setIsSignUp((prev) => !prev);
    };

    const handleLogIn = (event) => {
        event.preventDefault(); // Evita la recarga de la página
        const email = event.target[0].value;
        const password = event.target[1].value;

        fetch("http://localhost:8000/login", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ email, password }),
        })
            .then((response) => response.json())
            .then((data) => {
                console.log("Success:", data);
                if (data.success) {
                    navigate("/home");
                } else {
                    alert("Login failed");
                }
            })
            .catch((error) => {
                console.error("Error:", error);
            });
    };

    const handleSignUp = (event) => {
        event.preventDefault(); // Evita la recarga de la página
        const name = event.target[0].value;
        const email = event.target[1].value;
        const password = event.target[2].value;


        fetch("http://localhost:8000/register", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ name, email, password }),
        })
            .then((response) => response.json())
            .then((data) => {
                console.log("Success:", data);
                if (data.success) {
                    setIsSignUp(false);
                } else {
                    alert("Register failed");
                }
            })
            .catch((error) => {
                console.error("Error:", error);
            });
    };

    return (
        <div className={`auth-card-container ${isSignUp ? "auth-card-active" : ""}`}>
            <div className="auth-card-form-container auth-card-sign-up">
                <form onSubmit={handleSignUp}>
                    <h1>Registrate</h1>
                    <input type="text" placeholder="Name" />
                    <input type="email" placeholder="Email" />
                    <input type="password" placeholder="Password" />
                    <button type="submit">Registrarse</button>
                </form>
            </div>
            <div className="auth-card-form-container auth-card-sign-in">
                <form onSubmit={handleLogIn}>
                    <h1>Inicia Sesión</h1>
                    <input type="email" placeholder="Email" />
                    <input type="password" placeholder="Password" />
                    <a className="auth-card-link" href="/forgot-password">¿Has olvidado la contraseña?</a>
                    <button type="submit">Iniciar Sesión</button>
                </form>
            </div>
            <div className="auth-card-toggle-container">
                <div className="auth-card-toggle">
                    <div className="auth-card-toggle-panel auth-card-toggle-left">
                        <h1>¡Bienvenido de nuevo!</h1>
                        <p>Introduzca sus datos personales para utilizar todas las funciones del sitio web</p>
                        <button
                            className="auth-card-hidden"
                            id="login"
                            onClick={toggleForm}
                        >
                            Iniciar Sesión
                        </button>
                    </div>
                    <div className="auth-card-toggle-panel auth-card-toggle-right">
                        <h1>¡Hola, Amigo!</h1>
                        <p>Regístrese con sus datos personales para utilizar todas las funciones del sitio web</p>
                        <button
                            className="auth-card-hidden"
                            id="register"
                            onClick={toggleForm}
                        >
                            Registrarse
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default AuthCard;
