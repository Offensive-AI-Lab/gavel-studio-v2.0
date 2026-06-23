// Login is the entry point for every authenticated user, so its happy
// path and a few common failure modes need to stay green:
//   * the form submits credentials to loginUser()
//   * a 200 response stores token + user in localStorage
//   * failures surface a readable error message — both string and array
//     `detail` shapes from FastAPI must render
//   * the form does not navigate on failure

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

const navigateMock = vi.fn();
vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual('react-router-dom');
    return { ...actual, useNavigate: () => navigateMock };
});

vi.mock('../../src/api', () => ({
    loginUser: vi.fn(),
    syncLibrary: vi.fn(() => Promise.resolve()),
}));

import { loginUser } from '../../src/api';
import Login from '../../src/pages/Login';

const renderLogin = () => render(
    <MemoryRouter>
        <Login />
    </MemoryRouter>,
);

const fillAndSubmit = ({ email = 'a@b.c', password = 'pw' } = {}) => {
    fireEvent.change(screen.getByPlaceholderText(/Email/i), { target: { name: 'email', value: email } });
    fireEvent.change(screen.getByPlaceholderText(/Password/i), { target: { name: 'password', value: password } });
    fireEvent.click(screen.getByRole('button', { name: /Sign in/i }));
};


describe('Login page', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        navigateMock.mockClear();
    });

    it('renders the email + password fields and submit button', () => {
        renderLogin();
        expect(screen.getByPlaceholderText(/Email/i)).toBeInTheDocument();
        expect(screen.getByPlaceholderText(/Password/i)).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /Sign in/i })).toBeInTheDocument();
    });

    it('on success: stores token + user and navigates to /workspace', async () => {
        loginUser.mockResolvedValueOnce({
            data: { token: 'tok-123', user_id: 7, email: 'a@b.c' },
        });
        renderLogin();
        fillAndSubmit();

        await waitFor(() => {
            expect(sessionStorage.getItem('token')).toBe('tok-123');
        });
        const stored = JSON.parse(sessionStorage.getItem('user'));
        expect(stored.user_id).toBe(7);
        await waitFor(() => expect(navigateMock).toHaveBeenCalledWith('/workspace'));
    });

    it('renders a string-detail error from the API', async () => {
        loginUser.mockRejectedValueOnce({
            response: { data: { detail: 'Wrong password' } },
        });
        renderLogin();
        fillAndSubmit();
        await waitFor(() => {
            expect(screen.getByText(/Wrong password/)).toBeInTheDocument();
        });
        // No navigation on failure.
        expect(navigateMock).not.toHaveBeenCalled();
        // Token must NOT be set on failure.
        expect(sessionStorage.getItem('token')).toBeNull();
    });

    it('renders a default error message when the API returns no detail', async () => {
        loginUser.mockRejectedValueOnce(new Error('network'));
        renderLogin();
        fillAndSubmit();
        await waitFor(() => {
            expect(screen.getByText(/Invalid credentials/)).toBeInTheDocument();
        });
    });

    it('renders an array-detail error by joining the messages', async () => {
        // FastAPI validation errors come as `detail: [{ msg: 'too short' }, ...]`.
        loginUser.mockRejectedValueOnce({
            response: { data: { detail: [{ msg: 'too short' }, { msg: 'bad email' }] } },
        });
        renderLogin();
        fillAndSubmit();
        await waitFor(() => {
            // Both messages should appear in the joined string.
            expect(screen.getByText(/too short/)).toBeInTheDocument();
            expect(screen.getByText(/bad email/)).toBeInTheDocument();
        });
    });
});
