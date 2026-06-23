import React from 'react';
import Swal from 'sweetalert2';
import { renderToStaticMarkup } from 'react-dom/server';
import api, { getUserCEs } from '../api';
import PipelineModal from '../components/PipelineModal/PipelineModal';

const h = React.createElement;
const renderPipelineHtml = (props) => renderToStaticMarkup(h(PipelineModal, props));

const showPipelineMessage = async ({
    icon,
    title,
    subtitle,
    variant = 'light',
    confirmButtonColor = '#8b5cf6',
}) => Swal.fire({
    title: undefined,
    html: renderPipelineHtml({
        icon,
        title,
        subtitle,
        variant,
        showSpinner: false,
    }),
    confirmButtonColor,
});

/**
 * CEService: Handles logic for adding/removing Cognitive Elements.
 */
export const handleAddCEFlow = async (userId, currentRules, ruleIndex, updateState) => {
    const rule = currentRules[ruleIndex];
    const setupId = rule.setup_id;
    
    // 1. FILTER LOGIC: Get IDs of CEs already in this rule setup
    const existingCeIds = rule.active_ces.map(ce => ce.ce_id);

    const { value: result } = await Swal.fire({
        title: 'Add Cognitive Element',
        icon: 'question',
        showConfirmButton: true,
        showDenyButton: true,
        showCancelButton: true,
        confirmButtonText: 'Select Existing',
        denyButtonText: 'Create New (Hand)',
        cancelButtonText: 'Cancel',
        confirmButtonColor: '#2563eb',
        denyButtonColor: '#8b5cf6',
        cancelButtonColor: '#6b7280'
    });

    if (result === true) {
        const res = await getUserCEs(userId);
        const allCes = res.data.ces || res.data || [];

        // 2. EXCLUDE EXISTING: Only show CEs NOT already on the card
        const availableCes = allCes.filter(ce => !existingCeIds.includes(ce.ce_id));

        if (availableCes.length === 0) {
            return showPipelineMessage({
                icon: 'ℹ️',
                title: 'Nothing To Add',
                subtitle: 'All available elements are already in this rule.',
            });
        }

        const options = {};
        availableCes.forEach(ce => { options[ce.ce_id] = ce.name; });

        const { value: ceId } = await Swal.fire({
            title: 'Select CE',
            input: 'select',
            inputOptions: options,
            showCancelButton: true
        });

        if (ceId) {
            try {
                await api.post(`/rules/setup/${setupId}/link-ce`, { ce_id: ceId });
                updateLocalUI(currentRules, ruleIndex, options[ceId], ceId, updateState);
            } catch {
                await showPipelineMessage({
                    icon: '⛔',
                    title: 'Link Failed',
                    subtitle: 'Could not link the selected CE to this rule.',
                    confirmButtonColor: '#ef4444',
                });
            }
        }
    } else if (result === false) {
        const { value: name } = await Swal.fire({ title: 'Create CE', input: 'text', showCancelButton: true });
        if (name) {
            try {
                const res = await api.post(`/rules/setup/${setupId}/create-ce`, { name, user_id: userId });
                updateLocalUI(currentRules, ruleIndex, name, res.data.ce_id, updateState);
            } catch {
                await showPipelineMessage({
                    icon: '⛔',
                    title: 'Creation Failed',
                    subtitle: 'Could not create and attach a new CE.',
                    confirmButtonColor: '#ef4444',
                });
            }
        }
    }
};

export const handleRemoveCEFlow = async (setupId, ceId, ceName, currentRules, ruleIndex, updateState) => {
    try {
        await api.delete(`/rules/setup/${setupId}/ce/${ceId}`);
        const newRules = [...currentRules];
        
        // Remove from array
        newRules[ruleIndex].active_ces = newRules[ruleIndex].active_ces.filter(c => c.ce_id !== ceId);
        
        // 3. REBUILD Logic String: Re-joins the remaining names correctly
        const remaining = newRules[ruleIndex].active_ces.map(c => c.name);
        newRules[ruleIndex].predicate = remaining.length > 0 
            ? `IF ${remaining.join(' AND ')} THEN BLOCK` 
            : "IF TRUE THEN BLOCK";

        updateState(newRules);
    } catch {
        await showPipelineMessage({
            icon: '⛔',
            title: 'Remove Failed',
            subtitle: 'Failed to remove CE from this rule.',
            confirmButtonColor: '#ef4444',
        });
    }
};

const updateLocalUI = (rules, index, name, id, updateState) => {
    const newRules = [...rules];
    newRules[index].active_ces.push({ name, ce_id: id });
    
    // Clean string rebuild
    const ceNames = newRules[index].active_ces.map(c => c.name);
    newRules[index].predicate = `IF ${ceNames.join(' AND ')} THEN BLOCK`;
    
    updateState(newRules);
};