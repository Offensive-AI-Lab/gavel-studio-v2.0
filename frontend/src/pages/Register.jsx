import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { registerUser } from '../api';
import { FiUser, FiMail, FiLock, FiAlertCircle, FiShield, FiCheck } from 'react-icons/fi';
import AuthAside from '../components/AuthAside';
import '../css/auth.css';

const Register = () => {
    const navigate = useNavigate();
    const [formData, setFormData] = useState({ username: '', email: '', password: '' });
    const [error, setError] = useState('');
    const [isSuccess, setIsSuccess] = useState(false);
    const [submitting, setSubmitting] = useState(false);

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setSubmitting(true);
        try {
            await registerUser(formData);
            setIsSuccess(true);
            setTimeout(() => { navigate('/login'); }, 2000);
        } catch (err) {
            const detail = err.response?.data?.detail;
            let message = "Registration failed";
            if (typeof detail === 'string') {
                message = detail;
            } else if (Array.isArray(detail) && detail.length > 0) {
                message = detail.map(e => e.msg || JSON.stringify(e)).join(', ');
            }
            setError(message);
            setSubmitting(false);
        }
    };

    // Inline success state — replaces the legacy <Message/> component.
    // Same gradient-tile chrome as the brand logo above so the screen
    // doesn't flash to a different visual language on success.
    if (isSuccess) {
        return (
            <div className="auth-page">
                <div className="auth-shell">
                    <AuthAside />
                    <div className="auth-card">
                        <div className="auth-header">
                            <div className="auth-logo" style={{ background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)' }}>
                                <FiCheck size={28} />
                            </div>
                            <h1 className="auth-title">Account created</h1>
                            <p className="auth-subtitle">Redirecting you to sign in…</p>
                        </div>
                        <div className="auth-footer">
                            Take me there now<Link to="/login">Proceed to Login</Link>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="auth-page">
            <div className="auth-shell">
                <AuthAside />
                <div className="auth-card">
                    <div className="auth-header">
                        <div className="auth-logo"><FiShield size={28} /></div>
                        <h1 className="auth-title">Create an account</h1>
                        <p className="auth-subtitle">Start securing your AI models in minutes.</p>
                    </div>

                    {error && (
                        <div className="auth-error">
                            <FiAlertCircle /> {error}
                        </div>
                    )}

                    <form className="auth-form" onSubmit={handleSubmit}>
                        <div className="input-group">
                            <FiUser className="input-icon" />
                            <input
                                name="username"
                                placeholder="Username"
                                onChange={handleChange}
                                required
                                maxLength={30}
                                autoComplete="username"
                            />
                        </div>
                        {/* Permanence notice — usernames are immutable once set.
                          * Other users will see this name on every rule and CE
                          * you publish. The backend lowercases it on save
                          * regardless of how you type it (Abc → abc), so two
                          * users can't both register "Alice" and "alice". */}
                        <p style={{
                            margin: '-8px 4px 4px 4px',
                            fontSize: '0.78rem',
                            color: '#94a3b8',
                            lineHeight: 1.4,
                        }}>
                            Your username is <strong style={{ color: '#cbd5e1' }}>permanent</strong> — you can't change it later. Other users will see it on everything you publish.
                        </p>
                        <div className="input-group">
                            <FiMail className="input-icon" />
                            <input
                                name="email"
                                type="email"
                                placeholder="Email address"
                                onChange={handleChange}
                                required
                                maxLength={254}
                                autoComplete="email"
                            />
                        </div>
                        <div className="input-group">
                            <FiLock className="input-icon" />
                            <input
                                name="password"
                                type="password"
                                placeholder="Password (8+ characters)"
                                onChange={handleChange}
                                required
                                minLength={8}
                                maxLength={128}
                                autoComplete="new-password"
                            />
                        </div>

                        <button type="submit" className="auth-btn" disabled={submitting}>
                            {submitting ? 'Creating account…' : 'Create account'}
                        </button>
                    </form>

                    <div className="auth-footer">
                        Already have an account?<Link to="/login">Sign in</Link>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Register;
