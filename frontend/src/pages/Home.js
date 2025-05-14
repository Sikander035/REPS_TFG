import React, { useState, useEffect } from 'react';
import Navbar from '../components/Navbar';
import Footer from '../components/Footer';
import ExerciseCard from '../components/ExerciseCard';

const Home = () => {
    // Estado para almacenar los grupos musculares y sus ejercicios
    const [muscleGroups, setMuscleGroups] = useState({});

    // Función para obtener los ejercicios del servidor
    async function fetch_exercises() {
        try {
            const response = await fetch('http://localhost:8000/exercises');
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error fetching exercises:', error);
            return [];
        }
    }

    // Función para organizar los ejercicios por grupos musculares
    async function get_exercises() {
        const exercises = await fetch_exercises(); // Esperar a que los datos sean obtenidos
        let muscleGroups = {};

        exercises.forEach(exercise => {
            const { muscle_group } = exercise;
            // Si el grupo muscular no existe, inicializarlo como un array vacío
            if (!muscleGroups[muscle_group]) {
                muscleGroups[muscle_group] = [];
            }
            // Agregar el ejercicio al grupo muscular correspondiente
            muscleGroups[muscle_group].push(exercise);
        });

        return muscleGroups;
    }

    // Cargar los ejercicios al montar el componente
    useEffect(() => {
        async function loadExercises() {
            const data = await get_exercises();
            setMuscleGroups(data); // Guardar los datos en el estado
        }
        loadExercises();
    }, []); // Solo se ejecuta una vez al montar el componente

    return (
        <div className='page-container'>
            <Navbar />
            <div className='main-content'>
                {Object.keys(muscleGroups).length === 0 ? (
                    <></>
                ) : (
                    Object.keys(muscleGroups).map(muscleGroup => (
                        <div key={muscleGroup}>
                            <h1 className='section-title'>{muscleGroup}</h1>
                            <div className='muscle-group-exercises'>
                                {muscleGroups[muscleGroup].map(exercise => (
                                    <ExerciseCard
                                        key={exercise.name}
                                        name={exercise.name}
                                        imagePath={exercise.image_path}
                                        description={exercise.short_description}
                                    />
                                ))}
                            </div>
                        </div>
                    ))
                )}
            </div>
            <Footer />
        </div>
    );
};

export default Home;
