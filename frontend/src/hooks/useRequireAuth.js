import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

/**
 * Redirects to /login if no user is stored in sessionStorage (per-tab).
 * Returns the parsed user object (or null if not authenticated).
 */
const useRequireAuth = () => {
    const navigate = useNavigate();
    const user = JSON.parse(sessionStorage.getItem('user'));

    useEffect(() => {
        if (!user) {
            navigate('/login');
        }
    }, [navigate, user]);

    return user;
};

export default useRequireAuth;
