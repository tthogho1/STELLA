package com.example.billing;

import org.springframework.transaction.annotation.Transactional;

class BillingService {

    private final OrderRepository orderRepo;
    private final PaymentGateway paymentGateway;
    private final AuditLog auditLog;

    @Transactional
    public Receipt settle(String orderId, boolean force) {
        Order order = orderRepo.findById(orderId).orElseThrow(OrderMissingException::new);
        if (!force && order.isLocked()) {
            throw new OrderLockedException(orderId);
        }
        auditLog.record("settle", orderId);
        paymentGateway.charge(order.getTotal());
        order.markSettled();
        orderRepo.save(order);
        return receiptFactory.build(order);
    }

    public void cancel(String orderId) {
        Order order = orderRepo.findById(orderId).orElseThrow();
        order.markCancelled();
        orderRepo.save(order);
    }
}
