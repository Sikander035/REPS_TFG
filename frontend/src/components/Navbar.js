import React from 'react';
import { NavLink } from 'react-router-dom';
import UserProfile from './UserProfile';

const Navbar = () => {
    return (
        <div className='navbar'>
            <h1 className='navbar-title'>REPS</h1>
            <nav className='navbar-links'>
                <ul>
                    <li>
                        <NavLink
                            to='/home'
                            className={({ isActive }) => isActive ? 'active' : ''}
                        >
                            Home
                        </NavLink>
                    </li>
                    <li>
                        <NavLink
                            to='/about'
                            className={({ isActive }) => isActive ? 'active' : ''}
                        >
                            About
                        </NavLink>
                    </li>
                    <li>
                        <NavLink
                            to='/faq'
                            className={({ isActive }) => isActive ? 'active' : ''}
                        >
                            FAQ
                        </NavLink>
                    </li>
                </ul>
            </nav>
            <UserProfile />
        </div>
    );
};

export default Navbar;
