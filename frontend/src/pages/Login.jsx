import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { loginUser, syncLibrary } from '../api';
import { FiMail, FiLock, FiAlertCircle, FiShield } from 'react-icons/fi';
import AuthAside from '../components/AuthAside';
import '../css/auth.css';

const Login = () => {
    const [formData, setFormData] = useState({ email: '', password: '' });
    const [error, setError] = useState('');
    const [submitting, setSubmitting] = useState(false);
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setSubmitting(true);

        try {
            const response = await loginUser(formData);

            // Save token and user data in sessionStorage (NOT localStorage):
            // sessionStorage is scoped per browser tab, so two tabs on the same
            // machine can be signed into different accounts at once. localStorage
            // is shared across all tabs of an origin, which let a login in one
            // tab clobber the session in another. Tradeoff: closing a tab ends
            // that tab's session (no cross-restart persistence).
            sessionStorage.setItem('token', response.data.token);
            sessionStorage.setItem('user', JSON.stringify(response.data));

            // Fire-and-forget the library sync so login isn't blocked by an
            // HF round-trip. The server-startup bootstrap already kicks off
            // a sync_library() on boot, so by the time the user logs in the
            // cache is usually warm and this is a near no-op via the
            // manifest_hash short-circuit.
            syncLibrary()
                .catch((syncErr) => console.warn('Library sync at login failed; proceeding:', syncErr));

            navigate('/workspace');
        } catch (err) {
            const detail = err.response?.data?.detail;
            let message = "Invalid credentials";
            if (typeof detail === 'string') {
                message = detail;
            } else if (Array.isArray(detail) && detail.length > 0) {
                message = detail.map(e => e.msg || JSON.stringify(e)).join(', ');
            }
            setError(message);
            setSubmitting(false);
        }
    };

    return (
        <div className="auth-page">
            <div className="auth-shell">
                <AuthAside />
                <div className="auth-card">
                    <div className="auth-header">
                        <div className="auth-logo"><FiShield size={28} /></div>
                        <h1 className="auth-title">Welcome back</h1>
                        <p className="auth-subtitle">Sign in to your Gavel account.</p>
                    </div>

                    {error && (
                        <div className="auth-error">
                            <FiAlertCircle /> {error}
                        </div>
                    )}

                    <form className="auth-form" onSubmit={handleSubmit}>
                        <div className="input-group">
                            <FiMail className="input-icon" />
                            <input
                                name="email"
                                type="email"
                                placeholder="Email address"
                                value={formData.email}
                                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                                required
                                autoComplete="email"
                            />
                        </div>
                        <div className="input-group">
                            <FiLock className="input-icon" />
                            <input
                                name="password"
                                type="password"
                                placeholder="Password"
                                value={formData.password}
                                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                                required
                                autoComplete="current-password"
                            />
                        </div>
                        <button type="submit" className="auth-btn" disabled={submitting}>
                            {submitting ? 'Signing in…' : 'Sign in'}
                        </button>
                    </form>

                    <div className="auth-footer">
                        Don't have an account?<Link to="/register">Sign up</Link>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Login;
