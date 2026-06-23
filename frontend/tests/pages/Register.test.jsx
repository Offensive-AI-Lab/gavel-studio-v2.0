// Register flow:
//   * happy path: submit -> registerUser called with the form data ->
//     <Message> success view replaces the form -> navigates to /login
//     after a short delay
//   * failure path: error message renders, form remains visible
//   * detail can be string OR array (FastAPI validation errors)

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

const navigateMock = vi.fn();
vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual('react-router-dom');
    return { ...actual, useNavigate: () => navigateMock };
});

vi.mock('../../src/api', () => ({ registerUser: vi.fn() }));

import { registerUser } from '../../src/api';
import Register from '../../src/pages/Register';

const renderRegister = () => render(
    <MemoryRouter>
        <Register />
    </MemoryRouter>,
);

const fillAll = () => {
    fireEvent.change(screen.getByPlaceholderText('Username'), { target: { name: 'username', value: 'me' } });
    fireEvent.change(screen.getByPlaceholderText('Email address'), { target: { name: 'email', value: 'a@b.c' } });
    fireEvent.change(screen.getByPlaceholderText(/Password/i), { target: { name: 'password', value: 'pw345678' } });
};


describe('Register page', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        navigateMock.mockClear();
        // Defensive: a previous test may have crashed before its
        // `finally { vi.useRealTimers() }` ran, leaving the timer mock
        // in place and making waitFor() hang here.
        vi.useRealTimers();
    });

    it('renders all required input fields', () => {
        renderRegister();
        expect(screen.getByPlaceholderText('Username')).toBeInTheDocument();
        expect(screen.getByPlaceholderText('Email address')).toBeInTheDocument();
        expect(screen.getByPlaceholderText(/Password/i)).toBeInTheDocument();
        // Form intentionally has NO firstname / lastname fields anymore.
        expect(screen.queryByPlaceholderText(/First Name/i)).not.toBeInTheDocument();
        expect(screen.queryByPlaceholderText(/Last Name/i)).not.toBeInTheDocument();
        expect(screen.getByRole('button', { name: /Create account/i })).toBeInTheDocument();
    });

    it('on success: calls registerUser, shows the Message component, navigates to /login after the timeout', async () => {
        // Use real timers up front so waitFor's internal polling works, then
        // wait for the actual 2-second setTimeout to fire. Faking timers up
        // front fights waitFor and causes spurious 5s test timeouts.
        registerUser.mockResolvedValueOnce({ data: { ok: true } });
        renderRegister();
        fillAll();
        fireEvent.click(screen.getByRole('button', { name: /Create account/i }));

        await waitFor(() => {
            // Form payload no longer includes firstname / lastname; just
            // the three fields that are still in the UI.
            expect(registerUser).toHaveBeenCalledWith(expect.objectContaining({
                username: 'me', email: 'a@b.c', password: 'pw345678',
            }));
        });
        // Success state replaces the form. Title text is "Account created"
        // (mid-cap "C" in the new layout).
        await waitFor(() => {
            expect(screen.getByText(/Account created/i)).toBeInTheDocument();
        });
        // The 2-second navigation timer fires for real — give waitFor extra
        // room than the default 1s so we don't flake on a slow CI runner.
        await waitFor(
            () => expect(navigateMock).toHaveBeenCalledWith('/login'),
            { timeout: 3000 },
        );
    });

    it('on failure: renders the error string and keeps the form visible', async () => {
        registerUser.mockRejectedValueOnce({ response: { data: { detail: 'Username taken' } } });
        renderRegister();
        fillAll();
        fireEvent.click(screen.getByRole('button', { name: /Create account/i }));

        await waitFor(() => {
            expect(screen.getByText(/Username taken/)).toBeInTheDocument();
        });
        // Form is still there.
        expect(screen.getByPlaceholderText('Username')).toBeInTheDocument();
        expect(navigateMock).not.toHaveBeenCalled();
    });

    it('on failure: array-detail validation errors are joined and shown', async () => {
        registerUser.mockRejectedValueOnce({
            response: { data: { detail: [{ msg: 'email invalid' }, { msg: 'password too short' }] } },
        });
        renderRegister();
        fillAll();
        fireEvent.click(screen.getByRole('button', { name: /Create account/i }));

        await waitFor(() => {
            expect(screen.getByText(/email invalid/)).toBeInTheDocument();
            expect(screen.getByText(/password too short/)).toBeInTheDocument();
        });
    });

    it('on failure with no detail: shows the default error message', async () => {
        registerUser.mockRejectedValueOnce(new Error('boom'));
        renderRegister();
        fillAll();
        fireEvent.click(screen.getByRole('button', { name: /Create account/i }));
        await waitFor(() => {
            expect(screen.getByText(/Registration failed/i)).toBeInTheDocument();
        });
    });
});
