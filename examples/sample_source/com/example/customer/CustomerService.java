package com.example.customer;

class CustomerService {

    public CustomerResponse register(CustomerRequest request) {
        validator.validate(request);
        if (customerRepository.existsByCustomerNo(request.customerNo())) {
            throw new DuplicateCustomerException();
        }
        notificationClient.sendWelcome(request.email());
        return mapper.toResponse(customerRepository.save(entity));
    }
}
